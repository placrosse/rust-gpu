use rustc_data_structures::fx::FxHashMap;
use smallvec::SmallVec;
use spirt::func_at::{FuncAt, FuncAtMut};
use spirt::transform::InnerInPlaceTransform;
use spirt::visit::InnerVisit;
use spirt::{
    scalar, spv, vector, Const, ConstDef, ConstKind, Context, ControlNode, ControlNodeDef,
    ControlNodeKind, ControlNodeOutputDecl, ControlRegion, ControlRegionDef,
    ControlRegionInputDecl, DataInst, DataInstDef, DataInstFormDef, DataInstKind,
    EntityOrientedDenseMap, FuncDefBody, SelectionKind, Type, TypeKind, Value,
};
use std::convert::{TryFrom, TryInto};
use std::{iter, mem, slice};

use super::{ReplaceValueWith, VisitAllControlRegionsAndNodes};
use std::collections::hash_map::Entry;
use std::rc::Rc;

// HACK(eddyb) this exists only because the relevant changes ended up not being
// necessary in the end, but they might still be useful in the future.
const PREDICATION_FOR_SELECT_PER_CASE_OUTPUTS: bool = false;

/// Apply "reduction rules" to `func_def_body`, replacing (pure) computations
/// with one of their inputs or a constant (e.g. `x + 0 => x` or `1 + 2 => 3`),
/// and at most only adding more `Select` outputs/`Loop` state (where necessary)
/// but never any new instructions (unlike e.g. LLVM's instcombine).
pub(crate) fn reduce_in_func(cx: &Context, func_def_body: &mut FuncDefBody) {
    let wk = &super::SpvSpecWithExtras::get().well_known;

    let nop_inst_form = cx.intern(DataInstFormDef {
        kind: DataInstKind::SpvInst(wk.OpNop.into(), spv::InstLowering::default()),
        output_types: [].into_iter().collect(),
    });

    let mut parent_map = ParentMap::new(func_def_body);

    // FIXME(eddyb) entity-keyed dense maps might be better for performance,
    // but would require separate maps for separate `Value` cases.
    let mut value_replacements = FxHashMap::default();

    let mut reduction_cache = FxHashMap::default();

    // HACK(eddyb) this is an annoying workaround for iterator invalidation
    // (SPIR-T iterators don't cope well with the underlying data changing).
    //
    // FIXME(eddyb) replace SPIR-T `FuncAtMut<EntityListIter<T>>` with some
    // kind of "list cursor", maybe even allowing removal during traversal.
    let mut flattened_visit_events = vec![];

    // HACK(eddyb) reusable buffer for duplicating `Block` instruction lists
    // (as a workaround for iterator invalidation, see above).
    let mut tmp_block_insts = vec![];

    #[derive(Copy, Clone)]
    enum VisitEvent {
        EnterControlRegion(ControlRegion),
        ExitControlRegion(ControlRegion),
        EnterControlNode(ControlNode),
        ExitControlNode(ControlNode),
    }

    let mut iterations = 0;
    let start = std::time::Instant::now();
    loop {
        if iterations >= 1000 {
            // FIXME(eddyb) maybe attach a SPIR-T diagnostic instead?
            eprintln!(
                "[WARNING] spirt_passes::reduce: giving up on fixpoint after {iterations} iterations (took {:?})",
                start.elapsed()
            );
            break;
        }
        iterations += 1;

        let old_value_replacements_len = value_replacements.len();

        // HACK(eddyb) auto-rebuild reified visit whenever empty, to allow easy
        // invalidation if needed (see also `flattened_visit_events` comment).
        if flattened_visit_events.is_empty() {
            let mut visitor = VisitAllControlRegionsAndNodes {
                state: flattened_visit_events,
                enter_control_region: |events: &mut Vec<_>, facr: FuncAt<'_, ControlRegion>| {
                    events.push(VisitEvent::EnterControlRegion(facr.position));
                },
                exit_control_region: |events: &mut Vec<_>, facr: FuncAt<'_, ControlRegion>| {
                    events.push(VisitEvent::ExitControlRegion(facr.position));
                },
                enter_control_node: |events: &mut Vec<_>, facn: FuncAt<'_, ControlNode>| {
                    events.push(VisitEvent::EnterControlNode(facn.position));
                },
                exit_control_node: |events: &mut Vec<_>, facn: FuncAt<'_, ControlNode>| {
                    events.push(VisitEvent::ExitControlNode(facn.position));
                },
            };
            func_def_body.inner_visit_with(&mut visitor);
            flattened_visit_events = visitor.state;
        }

        // HACK(eddyb) always start with only the complete reductions from the
        // last iteration (as they can offer access to auxiliary definitions,
        // which avoids synthesizing new ones needlessly).
        reduction_cache.retain(|_, result| matches!(result, Some(Ok(_))));

        // FIXME(eddyb) consider alternatives (to these loops) that could be
        // "demand-driven" (recursing into use->def, instead of processing defs).
        //
        // FIXME(eddyb) it would be nice if `FuncAtMut` could be wrapped to track
        // `any_changes` itself, with immutable access (`.freeze()`) not setting it.
        let mut any_changes = false;
        for &visit_event in &flattened_visit_events {
            let func = func_def_body.at(());
            match visit_event {
                VisitEvent::EnterControlRegion(_) => {}
                VisitEvent::ExitControlRegion(control_region) => {
                    // HACK(eddyb) doing this early helps out `ExitControlNode`,
                    // which would otherwise need to do this itself, or wait
                    // for another iteration of the outer loop.
                    for output in &mut func_def_body.at_mut(control_region).def().outputs {
                        // FIXME(eddyb) run more of the reduction machinery,
                        // even if there is no operation applied to the value.
                        let mut v = *output;
                        while let Some(&new) = value_replacements.get(&v) {
                            any_changes = true;
                            v = new;
                        }
                        *output = v;
                    }
                }
                VisitEvent::EnterControlNode(control_node) => {
                    match &func.at(control_node).def().kind {
                        &ControlNodeKind::Block { insts } => {
                            // HACK(eddyb) iterator invalidation workaround
                            // (also see comment on `tmp_block_insts`).
                            assert!(tmp_block_insts.is_empty());
                            tmp_block_insts.extend(
                                func.at(insts)
                                    .into_iter()
                                    .map(|func_at_inst| func_at_inst.position),
                            );
                            for inst in tmp_block_insts.drain(..) {
                                // HACK(eddyb) applying `value_replacements` on
                                // the fly is pretty important to not breaking
                                // e.g. anything using `type_of`.
                                for input in &mut func_def_body.at_mut(inst).def().inputs {
                                    let mut v = *input;
                                    while let Some(&new) = value_replacements.get(&v) {
                                        any_changes = true;
                                        v = new;
                                    }
                                    *input = v;
                                }

                                let reduced =
                                    Reducible::try_from((cx, func_def_body.at(inst).def()))
                                        .ok()
                                        .and_then(|redu| {
                                            redu.try_reduce_to_value_or_incomplete(
                                                cx,
                                                func_def_body.at_mut(()),
                                                &parent_map,
                                                &mut reduction_cache,
                                            )
                                        });
                                match reduced {
                                    None => {}
                                    Some(Ok(v)) => {
                                        any_changes = true;
                                        value_replacements.insert(
                                            Value::DataInstOutput {
                                                inst,
                                                output_idx: 0,
                                            },
                                            v,
                                        );

                                        // HACK(eddyb) technically unnecessary (as `inst` gets
                                        // replaced with an `OpNop`), but it keeps things cleaner.
                                        match &mut func_def_body.control_nodes[control_node].kind {
                                            ControlNodeKind::Block { insts } => {
                                                insts.remove(inst, &mut func_def_body.data_insts);
                                            }
                                            _ => unreachable!(),
                                        };
                                        parent_map.data_inst_parent.remove(inst);

                                        // Replace the reduced `DataInstDef` itself with `OpNop`,
                                        // removing the ability to use its "name" as a value.
                                        *func_def_body.at_mut(inst).def() = DataInstDef {
                                            attrs: Default::default(),
                                            form: nop_inst_form,
                                            inputs: [].into_iter().collect(),
                                        };
                                    }
                                    Some(Err(Incomplete(redu))) => {
                                        if let Some(redu_data_inst_def) = redu.try_into_inst(cx) {
                                            any_changes = true;
                                            *func_def_body.at_mut(inst).def() = redu_data_inst_def;
                                        }
                                    }
                                }
                            }
                        }

                        ControlNodeKind::Select {
                            kind,
                            scrutinee,
                            cases,
                        } => {
                            // FIXME(eddyb) run more of the reduction machinery,
                            // even if there is no operation applied to the value.
                            let scrutinee = {
                                let mut v = *scrutinee;
                                while let Some(&new) = value_replacements.get(&v) {
                                    any_changes = true;
                                    v = new;
                                }
                                v
                            };

                            // HACK(eddyb) minimum viable "`Select` by constant"
                            // simplification, e.g. `if false {A} else {B} -> B`.
                            if let Value::Const(scrutinee) = scrutinee {
                                let case_consts = match kind {
                                    SelectionKind::BoolCond => &[scalar::Const::TRUE][..],
                                    SelectionKind::Switch { case_consts } => case_consts,
                                };
                                let taken_case = scrutinee
                                    .as_scalar(cx)
                                    .filter(|s| case_consts.iter().all(|c| c.ty() == s.ty()))
                                    .map(|s| {
                                        cases[case_consts
                                            .iter()
                                            .position(|c| c == s)
                                            .unwrap_or(case_consts.len())]
                                    });
                                if let Some(taken_case) = taken_case {
                                    any_changes = true;

                                    let ControlRegionDef {
                                        inputs,
                                        mut children,
                                        outputs,
                                    } = mem::take(func_def_body.at_mut(taken_case).def());

                                    assert_eq!(inputs.len(), 0);

                                    // Move every child of the taken case region, to just before
                                    // `control_node`, in its parent region (effectively replacing it).
                                    let parent_region =
                                        parent_map.control_node_parent[control_node];
                                    let parent_region_children = &mut func_def_body.control_regions
                                        [parent_map.control_node_parent[control_node]]
                                        .children;
                                    while let Some(case_child) = children.iter().first {
                                        children
                                            .remove(case_child, &mut func_def_body.control_nodes);
                                        parent_region_children.insert_before(
                                            case_child,
                                            control_node,
                                            &mut func_def_body.control_nodes,
                                        );
                                        parent_map.control_node_parent[case_child] = parent_region;
                                    }

                                    // HACK(eddyb) technically unnecessary (as `control_node` gets
                                    // replaced with an empty `Block`), but it keeps things cleaner.
                                    parent_region_children
                                        .remove(control_node, &mut func_def_body.control_nodes);
                                    parent_map.control_node_parent.remove(control_node);

                                    *func_def_body.at_mut(control_node).def() = ControlNodeDef {
                                        kind: ControlNodeKind::Block {
                                            insts: Default::default(),
                                        },
                                        outputs: Default::default(),
                                    };

                                    for (i, &v) in outputs.iter().enumerate() {
                                        value_replacements
                                            .entry(Value::ControlNodeOutput {
                                                control_node,
                                                output_idx: i.try_into().unwrap(),
                                            })
                                            .or_insert(v);
                                    }

                                    // FIXME(eddyb) `flattened_visit_events`
                                    // could be rebuilt in-place, and the dead
                                    // cases skipped through clever tracking,
                                    // to avoid bailing out so early.
                                    flattened_visit_events.clear();
                                    break;
                                }
                            }

                            let scrutinee_redu_op = match kind {
                                // HACK(eddyb) turn `if c {A} else {B}` into `if !c {B} else {A}`,
                                // with the expectation that `!c` will undo some other negation,
                                // e.g. `if d { false } else { true }`.
                                // NOTE(eddyb) endless cycling is avoided in `try_reduce` by
                                // refusing to generate new e.g. `ControlNode` outputs for
                                // `bool.not` (and only allow "true" reductions).
                                SelectionKind::BoolCond => {
                                    // FIXME(eddyb) try having an identity "reduction" op?
                                    Some(PureOp::BoolUnOp(scalar::BoolUnOp::Not))
                                }

                                // HACK(eddyb) turn `switch x { 0 => A, 1 => B, _ => ... }`
                                // into `if ... {B} else {A}`, when `x` ends up limited in `0..=1`,
                                // (such `switch`es come from e.g. `match`-ing enums w/ 2 variants)
                                // allowing us to bypass SPIR-T current (and temporary) lossiness
                                // wrt `_ => OpUnreachable` (i.e. we prove the default case can't
                                // be entered based on `x` not having values other than `0` or `1`)
                                //
                                // FIXME(eddyb) support more values than just `0..=1`.
                                SelectionKind::Switch { case_consts }
                                    if cases.len() == 3
                                        && case_consts.iter().enumerate().all(|(i, ct)| {
                                            ct.int_as_u32() == Some(u32::try_from(i).unwrap())
                                        }) =>
                                {
                                    Some(PureOp::IntToBool)
                                }

                                SelectionKind::Switch { .. } => None,
                            };

                            let (new_scrutinee_op, new_scrutinee, flipped) = scrutinee_redu_op
                                .and_then(|op| {
                                    let redu = Reducible {
                                        op,
                                        output_type: cx.intern(scalar::Type::Bool),
                                        input: scrutinee,
                                    };
                                    let reduced = redu.try_reduce_to_value_or_incomplete(
                                        cx,
                                        func_def_body.at_mut(()),
                                        &parent_map,
                                        &mut reduction_cache,
                                    );
                                    match reduced {
                                        Some(Ok(v)) => Some((op, v, false)),
                                        Some(Err(Incomplete(Reducible {
                                            op: PureOp::BoolUnOp(scalar::BoolUnOp::Not),
                                            input,
                                            ..
                                        }))) => Some((op, input, true)),
                                        _ => None,
                                    }
                                })
                                .map_or((None, scrutinee, false), |(op, v, flipped)| {
                                    (Some(op), v, flipped)
                                });

                            let func_at_control_node = func_def_body.at_mut(control_node);
                            match &mut func_at_control_node.def().kind {
                                ControlNodeKind::Select {
                                    kind,
                                    scrutinee,
                                    cases,
                                } => {
                                    *scrutinee = new_scrutinee;
                                    if let Some(op) = new_scrutinee_op {
                                        any_changes = true;
                                        let flipped = match (op, &*kind, &cases[..]) {
                                            (
                                                PureOp::IntToBool,
                                                SelectionKind::Switch { .. },
                                                &[case_0, case_1, _default],
                                            ) => {
                                                *kind = SelectionKind::BoolCond;
                                                *cases = [case_1, case_0].into_iter().collect();
                                                flipped
                                            }

                                            // FIXME(eddyb) the double flip is confusing.
                                            (
                                                PureOp::BoolUnOp(scalar::BoolUnOp::Not),
                                                SelectionKind::BoolCond,
                                                _,
                                            ) => !flipped,

                                            _ => unreachable!(),
                                        };
                                        if flipped {
                                            cases.swap(0, 1);
                                        }
                                    }
                                }
                                _ => unreachable!(),
                            }
                        }

                        ControlNodeKind::Loop { .. } => {
                            // HACK(eddyb) doing this early helps out `ExitControlNode`,
                            // which would otherwise need to do this itself, or wait
                            // for another iteration of the outer loop.
                            let initial_inputs =
                                match &mut func_def_body.at_mut(control_node).def().kind {
                                    ControlNodeKind::Loop { initial_inputs, .. } => initial_inputs,
                                    _ => unreachable!(),
                                };
                            for initial_input in initial_inputs {
                                // FIXME(eddyb) run more of the reduction machinery,
                                // even if there is no operation applied to the value.
                                let mut v = *initial_input;
                                while let Some(&new) = value_replacements.get(&v) {
                                    any_changes = true;
                                    v = new;
                                }
                                *initial_input = v;
                            }
                        }
                    }
                }
                VisitEvent::ExitControlNode(control_node) => {
                    let control_node_def = &func_def_body.control_nodes[control_node];
                    match &control_node_def.kind {
                        ControlNodeKind::Block { .. } => {}

                        ControlNodeKind::Select {
                            kind,
                            scrutinee,
                            cases,
                        } => {
                            if PREDICATION_FOR_SELECT_PER_CASE_OUTPUTS {
                                // FIXME(eddyb) DRY.
                                let case_consts = match kind {
                                    SelectionKind::BoolCond => {
                                        &[scalar::Const::TRUE, scalar::Const::FALSE][..]
                                    }
                                    SelectionKind::Switch { case_consts } => case_consts,
                                };

                                // HACK(eddyb) it's not uncommon to have chained
                                // conditionals with the same condition, and end up
                                // with overly generic values, when more specific
                                // ones can be known, thanks to the condition.
                                // FIXME(eddyb) can this degrade `try_reduce_select`'s
                                // ability to reuse a previous `Select`'s outputs?
                                // (maybe this should be attempted only if merging
                                // the per-case outputs together has failed)
                                for (&case, &scrutinee_expected) in cases.iter().zip(case_consts) {
                                    for output_idx in 0..control_node_def.outputs.len() {
                                        let per_case_output =
                                            func_def_body.control_regions[case].outputs[output_idx];
                                        if let Some(per_case_output_assuming_case) =
                                            try_reduce_predicated(
                                                cx,
                                                &parent_map,
                                                func_def_body.at(()),
                                                per_case_output,
                                                *scrutinee,
                                                scrutinee_expected,
                                            )
                                        {
                                            if per_case_output_assuming_case != per_case_output {
                                                any_changes = true;
                                                func_def_body.control_regions[case].outputs
                                                    [output_idx] = per_case_output_assuming_case;
                                            }
                                        }
                                    }
                                }
                            }

                            let func = func_def_body.at(());
                            for i in 0..u32::try_from(control_node_def.outputs.len()).unwrap() {
                                let output = Value::ControlNodeOutput {
                                    control_node,
                                    output_idx: i,
                                };
                                if let Entry::Vacant(entry) = value_replacements.entry(output) {
                                    let per_case_value = cases
                                        .iter()
                                        .map(|&case| func.at(case).def().outputs[i as usize]);
                                    if let Some(reduced) = try_reduce_select(
                                        cx,
                                        &parent_map,
                                        func,
                                        Some(control_node),
                                        kind,
                                        *scrutinee,
                                        per_case_value,
                                    ) {
                                        entry.insert(reduced);
                                    }
                                }
                            }
                        }

                        ControlNodeKind::Loop {
                            body,
                            initial_inputs,
                            repeat_condition,
                        } => {
                            let body = *body;

                            let loop_state_count = initial_inputs.len();

                            // FIXME(eddyb) run more of the reduction machinery,
                            // even if there is no operation applied to the value.
                            let repeat_condition = {
                                let mut v = *repeat_condition;
                                while let Some(&new) = value_replacements.get(&v) {
                                    v = new;
                                }
                                v
                            };

                            for i in 0..u32::try_from(loop_state_count).unwrap() {
                                let body_input = Value::ControlRegionInput {
                                    region: body,
                                    input_idx: i,
                                };
                                let mut body_output =
                                    func_def_body.at(body).def().outputs[i as usize];

                                // HACK(eddyb) take advantage of the loop body
                                // outputs being meant for the next iteration
                                // (only needed when `repeat_condition == true`).
                                if let Some(body_output_assuming_repeating) = try_reduce_predicated(
                                    cx,
                                    &parent_map,
                                    func_def_body.at(()),
                                    body_output,
                                    repeat_condition,
                                    scalar::Const::TRUE,
                                ) {
                                    if body_output_assuming_repeating != body_output {
                                        any_changes = true;
                                        func_def_body.at_mut(body).def().outputs[i as usize] =
                                            body_output_assuming_repeating;
                                        body_output = body_output_assuming_repeating;
                                    }
                                }

                                if body_output == body_input {
                                    let initial_input =
                                        match &func_def_body.at(control_node).def().kind {
                                            ControlNodeKind::Loop { initial_inputs, .. } => {
                                                initial_inputs[i as usize]
                                            }
                                            _ => unreachable!(),
                                        };
                                    value_replacements
                                        .entry(body_input)
                                        .or_insert(initial_input);
                                }
                            }

                            // HACK(eddyb) it's not uncommon for a loop body to
                            // end in a `Select` that simultaneously produces
                            // the repeat condition, loop body outputs (loop state),
                            // and its own separate outputs only used if the loop
                            // completes, i.e. on the last iteration, when the
                            // `repeat_condition` becomes `false`.
                            //
                            // FIXME(eddyb) how necessary is this after all?
                            if let Some(tail_node) =
                                func_def_body.at(body).def().children.iter().last
                            {
                                for i in 0..u32::try_from(
                                    func_def_body.at(tail_node).def().outputs.len(),
                                )
                                .unwrap()
                                {
                                    let tail_output = Value::ControlNodeOutput {
                                        control_node: tail_node,
                                        output_idx: i,
                                    };
                                    let tail_output_used_by_loop = repeat_condition == tail_output
                                        || func_def_body
                                            .at(body)
                                            .def()
                                            .outputs
                                            .contains(&tail_output);
                                    if tail_output_used_by_loop {
                                        continue;
                                    }

                                    // HACK(eddyb) ideally this could be handled
                                    // more generally, but that would require more
                                    // advanced/contextual predication systems
                                    // (region nesting might be sufficient?).
                                    let predicate = repeat_condition;
                                    let predicate_expected = scalar::Const::FALSE;
                                    match predicate {
                                        Value::ControlNodeOutput {
                                            control_node: p_node,
                                            output_idx: p_idx,
                                        } if p_node == tail_node => {
                                            let cases = match &func_def_body.control_nodes
                                                [tail_node]
                                                .kind
                                            {
                                                ControlNodeKind::Select { cases, .. } => cases,
                                                _ => unreachable!(),
                                            };
                                            for &case in cases {
                                                let per_case_outputs =
                                                    &func_def_body.at(case).def().outputs;
                                                let per_case_output = per_case_outputs[i as usize];
                                                if let Some(per_case_output_assuming_exiting) =
                                                    try_reduce_predicated(
                                                        cx,
                                                        &parent_map,
                                                        func_def_body.at(()),
                                                        per_case_output,
                                                        per_case_outputs[p_idx as usize],
                                                        predicate_expected,
                                                    )
                                                {
                                                    if per_case_output_assuming_exiting
                                                        != per_case_output
                                                    {
                                                        any_changes = true;
                                                        func_def_body.control_regions[case]
                                                            .outputs
                                                            [i as usize] =
                                                            per_case_output_assuming_exiting;
                                                    }
                                                }
                                            }
                                        }
                                        _ => {
                                            if let Some(tail_output_assuming_exiting) =
                                                try_reduce_predicated(
                                                    cx,
                                                    &parent_map,
                                                    func_def_body.at(()),
                                                    tail_output,
                                                    predicate,
                                                    predicate_expected,
                                                )
                                            {
                                                if tail_output_assuming_exiting != tail_output {
                                                    value_replacements
                                                        .entry(tail_output)
                                                        .or_insert(tail_output_assuming_exiting);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !any_changes && old_value_replacements_len == value_replacements.len() {
            break;
        }

        func_def_body.inner_in_place_transform_with(&mut ReplaceValueWith(|mut v| {
            let old = v;
            loop {
                match v {
                    Value::Const(_) => break,
                    _ => match value_replacements.get(&v) {
                        Some(&new) => v = new,
                        None => break,
                    },
                }
            }
            if v != old {
                // FIXME(eddyb) this is silly, but it should be impossible to reach this,
                // now that `value_replacements` is eagerly applied in the loop above,
                // at least when `flattened_visit_events` hasn't been reset mid-way.
                if false {
                    // FIXME(eddyb) this has issues with loops, which can replace
                    // the body inputs after the loop body is visited - maybe
                    // the solution is tracking lengths in `flattened_visit_events`,
                    // so traversal can restart to the start of the loop, or even
                    // skip across `Select`s replaced with a single case.
                    assert!(flattened_visit_events.is_empty());
                }
                any_changes = true;
                Some(v)
            } else {
                None
            }
        }));
    }
}

// FIXME(eddyb) maybe this kind of "parent map" should be provided by SPIR-T?
#[derive(Default)]
struct ParentMap {
    data_inst_parent: EntityOrientedDenseMap<DataInst, ControlNode>,
    control_node_parent: EntityOrientedDenseMap<ControlNode, ControlRegion>,
    control_region_parent: EntityOrientedDenseMap<ControlRegion, ControlNode>,
}

impl ParentMap {
    fn new(func_def_body: &FuncDefBody) -> Self {
        let mut visitor = VisitAllControlRegionsAndNodes {
            state: Self::default(),
            enter_control_region:
                |this: &mut Self, func_at_control_region: FuncAt<'_, ControlRegion>| {
                    for func_at_child_control_node in func_at_control_region.at_children() {
                        this.control_node_parent.insert(
                            func_at_child_control_node.position,
                            func_at_control_region.position,
                        );
                    }
                },
            exit_control_region: |_: &mut _, _| {},
            enter_control_node: |this: &mut Self, func_at_control_node: FuncAt<'_, ControlNode>| {
                let child_regions = match &func_at_control_node.def().kind {
                    &ControlNodeKind::Block { insts } => {
                        for func_at_inst in func_at_control_node.at(insts) {
                            this.data_inst_parent
                                .insert(func_at_inst.position, func_at_control_node.position);
                        }
                        &[][..]
                    }

                    ControlNodeKind::Select { cases, .. } => cases,
                    ControlNodeKind::Loop { body, .. } => slice::from_ref(body),
                };
                for &child_region in child_regions {
                    this.control_region_parent
                        .insert(child_region, func_at_control_node.position);
                }
            },
            exit_control_node: |_: &mut _, _| {},
        };
        func_def_body.inner_visit_with(&mut visitor);
        visitor.state
    }
}

/// If possible, find a single `Value` from `cases` (or even `scrutinee`),
/// which would always be a valid result for `Select(kind, scrutinee, cases)`,
/// regardless of which case gets (dynamically) taken.
fn try_reduce_select(
    cx: &Context,
    parent_map: &ParentMap,
    // FIXME(eddyb) come up with a better convention for this!
    func: FuncAt<'_, ()>,
    // HACK(eddyb) `None` means this is actually an `OpSelect`.
    select_control_node: Option<ControlNode>,
    kind: &SelectionKind,
    scrutinee: Value,
    cases: impl Iterator<Item = Value>,
) -> Option<Value> {
    let as_const = |v: Value| match v {
        Value::Const(ct) => Some(ct),
        _ => None,
    };

    // Rewrite `v` to assume `case_idx` was taken, i.e. replacing
    // `scrutinee` and values depending on it (on a best-effort basis).
    let specialize_for_case = |mut v: Value, case_idx: usize| {
        // FIXME(eddyb) might make sense to do actual recursion, and
        // even engage the full reduction machinery.
        if v == scrutinee {
            match kind {
                // FIXME(eddyb) is this wasteful? do bools need caching?
                SelectionKind::BoolCond => {
                    return Value::Const(
                        cx.intern([scalar::Const::TRUE, scalar::Const::FALSE][case_idx]),
                    );
                }
                SelectionKind::Switch { case_consts } => {
                    if let Some(&case_const) = case_consts.get(case_idx) {
                        return Value::Const(cx.intern(case_const));
                    }
                }
            }
        }

        // FIXME(eddyb) this overlaps a lot with `try_reduce_predicated`.
        if let Value::ControlNodeOutput {
            control_node,
            output_idx,
        } = v
        {
            match &func.at(control_node).def().kind {
                ControlNodeKind::Select {
                    kind: v_def_kind,
                    scrutinee: v_def_scrutinee,
                    cases: v_def_cases,
                } if *v_def_scrutinee == scrutinee => {
                    let equal_kind = match (v_def_kind, kind) {
                        (SelectionKind::BoolCond, SelectionKind::BoolCond) => true,
                        (
                            SelectionKind::Switch { case_consts: a },
                            SelectionKind::Switch { case_consts: b },
                        ) => a == b,
                        _ => false,
                    };
                    if equal_kind {
                        v = func.at(v_def_cases[case_idx]).def().outputs[output_idx as usize];
                    }
                }
                _ => {}
            }
        }

        v
    };

    // Ignore `undef`s, as they can be legally substituted with any other value.
    let mut first_undef = None;
    let mut non_undef_cases = cases.enumerate().filter(|&(_, case)| {
        let is_undef = as_const(case).map(|ct| &cx[ct].kind) == Some(&ConstKind::Undef);
        if is_undef && first_undef.is_none() {
            first_undef = Some(case);
        }
        !is_undef
    });
    // FIXME(eddyb) false positive (no pre-existing tuple, only multi-value `match`ing).
    #[allow(clippy::tuple_array_conversions)]
    let merged_value = match (
        non_undef_cases.next(),
        non_undef_cases.next(),
        non_undef_cases.next(),
    ) {
        (None, ..) => first_undef?,

        // `Select(c: bool, f(true), f(false))` can be replaced with just `f(c)`,
        // if some suitable `f` can be found, such as:
        // - the identity function, i.e. `Select(c, true, false) -> c`
        // - the output of another `Select(c, _, _)` used in only one case,
        //   e.g.: `Select(c, Select(c, x, y), y) -> Select(c, x, y)`
        //
        // FIXME(eddyb) technically this can work for `switch`, too, but
        // that may require buffering the values before inspecting them.
        (Some((0, x)), Some((1, y)), None) if x != y && matches!(kind, SelectionKind::BoolCond) => {
            assert!(first_undef.is_none());

            let specific_cases = [x, y];
            ([scrutinee].into_iter().chain(specific_cases)).find(|&candidate| {
                // A general `candidate` which agrees with each individual case
                // can fully replace the original specific values, provided it
                // does dominate the `Select` itself (which is checked later).
                specific_cases.iter().enumerate().all(|(i, &specific)| {
                    specific == candidate
                        || specialize_for_case(specific, i) == specialize_for_case(candidate, i)
                })
            })?
        }

        (Some((x_idx, x)), y, z) => {
            if (y.into_iter().chain(z).chain(non_undef_cases)).all(|(_, v)| v == x) {
                // HACK(eddyb) if the same value appears in two different
                // cases, it's definitely dominating the whole `Select`.
                if y.is_some() {
                    return Some(x);
                }

                // HACK(eddyb) only one non-`undef` case, specializing to avoid
                // losing path-dependent information e.g. `Select(c, c, undef)`
                // can be replaced with `c` or `true` (as it's `c | undef`).
                //
                // FIXME(eddyb) there has to be a better way to do this, esp.
                // as the `Select(c: bool, f(true), f(false))` transformation
                // (see comment higher above) prefers generalizing, while using
                // specialization to guarantee viability of some general form
                // (maybe the solution is to keep the most general form, and
                // compute the per-case specialization, using the former only
                // when the latter doesn't overlap between cases).
                specialize_for_case(x, x_idx)
            } else {
                return None;
            }
        }
    };

    let is_valid_outside_select = |x: Value| {
        let select_control_node = match select_control_node {
            Some(select_control_node) => select_control_node,

            // HACK(eddyb) `select_control_node = None` indicates an
            // `OpSelect`, which is always dominated by its inputs.
            None => return Some(()),
        };

        // HACK(eddyb) the `scrutinee` has to dominate the `Select` itself.
        if x == scrutinee {
            return Some(());
        }

        // In general, `x` dominating the `Select` is what would
        // allow lifting an use of it outside the `Select`.
        let mut region_defining_x = match x {
            // Constants are always valid.
            Value::Const(_) => return Some(()),

            Value::ControlRegionInput { region, .. } => region,
            Value::ControlNodeOutput { control_node, .. } => {
                *parent_map.control_node_parent.get(control_node)?
            }
            Value::DataInstOutput { inst, .. } => *parent_map
                .control_node_parent
                .get(*parent_map.data_inst_parent.get(inst)?)?,
        };

        while let Some(&parent_of_region_defining_x) =
            parent_map.control_region_parent.get(region_defining_x)
        {
            // Fast-reject: if `x` is defined immediately inside one of
            // `select_control_node`'s cases, it's not a dominator.
            if parent_of_region_defining_x == select_control_node {
                return None;
            }

            // HACK(eddyb) due to SSA semantics (instead of RVSDG regions),
            // the body of a `Loop` dominates everything the `Loop` dominates.
            match func.at(parent_of_region_defining_x).def().kind {
                ControlNodeKind::Loop { body, .. } if body == region_defining_x => {
                    region_defining_x = parent_map
                        .control_node_parent
                        .get(parent_of_region_defining_x)
                        .copied()?;
                    continue;
                }
                _ => {}
            }

            break;
        }

        // Since we know `x` is used inside the `Select`, this only
        // needs to check that `x` is defined in a region that the
        // `Select` is nested in, as the only other possibility is
        // that the `x` is defined inside the `Select` - that is,
        // one of `x` and `Select` always dominates the other.
        //
        // FIXME(eddyb) this could be more efficient with some kind
        // of "region depth" precomputation but a potentially-slower
        // check doubles as a sanity check, for now.
        let mut region_containing_select =
            *parent_map.control_node_parent.get(select_control_node)?;
        loop {
            if region_containing_select == region_defining_x {
                return Some(());
            }
            region_containing_select = *parent_map.control_node_parent.get(
                *parent_map
                    .control_region_parent
                    .get(region_containing_select)?,
            )?;
        }
    };

    is_valid_outside_select(merged_value).map(|()| merged_value)
}

/// Attempt to "simplify" `v` assuming `predicate == predicate_expected`
/// (i.e. disregarding values `v` can take in any other situations).
//
// FIXME(eddyb) how necessary is this after all?
fn try_reduce_predicated(
    cx: &Context,
    parent_map: &ParentMap,
    // FIXME(eddyb) come up with a better convention for this!
    func: FuncAt<'_, ()>,

    v: Value,
    predicate: Value,
    predicate_expected: scalar::Const,
) -> Option<Value> {
    match (v, predicate) {
        _ if v == predicate => Some(Value::Const(cx.intern(predicate_expected))),

        (Value::Const(_), _) => Some(v),
        (_, Value::Const(p)) => {
            let p = p
                .as_scalar(cx)
                .filter(|p| p.ty() == predicate_expected.ty())?;
            Some(if p == &predicate_expected {
                v
            } else {
                // FIXME(eddyb) probably inefficient, return an `enum` instead?
                Value::Const(cx.intern(ConstDef {
                    attrs: Default::default(),
                    ty: func.at(v).type_of(cx),
                    kind: ConstKind::Undef,
                }))
            })
        }

        // FIXME(eddyb) this prefers recursing on `predicate`, but it's possible
        // `v` is derived from something that depends on `predicate`.
        (
            _,
            Value::ControlNodeOutput {
                control_node: p_node,
                output_idx: p_idx,
            },
        ) => {
            match &func.at(p_node).def().kind {
                ControlNodeKind::Select {
                    kind,
                    scrutinee,
                    cases,
                } => {
                    // FIXME(eddyb) collecting only because of per-case `Option`.
                    let per_case_predicated: SmallVec<[_; 2]> = cases
                        .iter()
                        .map(|&case| {
                            // FIXME(eddyb) ideally predication on `scrutinee`
                            // would be also done here.
                            let per_case_outputs = &func.at(case).def().outputs;
                            let v = match v {
                                Value::ControlNodeOutput {
                                    control_node,
                                    output_idx,
                                } if control_node == p_node => {
                                    per_case_outputs[output_idx as usize]
                                }
                                _ => v,
                            };
                            try_reduce_predicated(
                                cx,
                                parent_map,
                                func,
                                v,
                                per_case_outputs[p_idx as usize],
                                predicate_expected,
                            )
                        })
                        .collect::<Option<_>>()?;
                    try_reduce_select(
                        cx,
                        parent_map,
                        func,
                        Some(p_node),
                        kind,
                        *scrutinee,
                        per_case_predicated.iter().copied(),
                    )
                }
                _ => unreachable!(),
            }
        }

        (
            Value::ControlNodeOutput {
                control_node: v_node,
                output_idx: v_idx,
            },
            _,
        ) if PREDICATION_FOR_SELECT_PER_CASE_OUTPUTS => {
            match &func.at(v_node).def().kind {
                ControlNodeKind::Select {
                    kind,
                    scrutinee,
                    cases,
                } => {
                    // FIXME(eddyb) DRY.
                    let case_consts = match kind {
                        SelectionKind::BoolCond => &[scalar::Const::TRUE, scalar::Const::FALSE][..],
                        SelectionKind::Switch { case_consts } => case_consts,
                    };

                    // FIXME(eddyb) collecting only because of per-case `Option`.
                    let per_case_predicated: SmallVec<[_; 2]> = cases
                        .iter()
                        .enumerate()
                        .map(|(case_idx, &case)| {
                            let per_case_outputs = &func.at(case).def().outputs;
                            // FIXME(eddyb) this is completely inscrutable, at
                            // least it should be traversing the DAG so it will
                            // terminate, but predication should be systematic.
                            let predicate = case_consts
                                .get(case_idx)
                                .and_then(|&scrutinee_expected| {
                                    try_reduce_predicated(
                                        cx,
                                        parent_map,
                                        func,
                                        predicate,
                                        *scrutinee,
                                        scrutinee_expected,
                                    )
                                })
                                .unwrap_or(predicate);
                            try_reduce_predicated(
                                cx,
                                parent_map,
                                func,
                                per_case_outputs[v_idx as usize],
                                predicate,
                                predicate_expected,
                            )
                        })
                        .collect::<Option<_>>()?;
                    try_reduce_select(
                        cx,
                        parent_map,
                        func,
                        Some(v_node),
                        kind,
                        *scrutinee,
                        per_case_predicated.iter().copied(),
                    )
                }
                _ => unreachable!(),
            }
        }

        (Value::DataInstOutput { inst, output_idx }, _)
            if PREDICATION_FOR_SELECT_PER_CASE_OUTPUTS =>
        {
            let try_reduce_predicated =
                |v| try_reduce_predicated(cx, parent_map, func, v, predicate, predicate_expected);

            let mut redu = Reducible::try_from((cx, func.at(inst).def())).ok()?;
            assert_eq!(output_idx, 0);

            loop {
                if let Some(predicated_input) = try_reduce_predicated(redu.input) {
                    redu.input = predicated_input;
                }
                match redu.try_reduce_shallow(cx, |v| func.at(v).type_of(cx))? {
                    ReductionStep::Complete(v) => return try_reduce_predicated(v),
                    ReductionStep::Partial(redu_next) => redu = redu_next,
                }
            }
        }

        _ => None,
    }
}

/// Pure operation that transforms one `Value` into another `Value`.
//
// FIXME(eddyb) move this elsewhere? also, how should binops etc. be supported?
// (one approach could be having a "focus input" that can be dynamic, with the
// other inputs being `Const`s, i.e. partially applying all but one input)
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum PureOp {
    BoolUnOp(scalar::BoolUnOp),
    IntUnOp(scalar::IntUnOp),

    /// Integer binary operation with a constant RHS (e.g. `x + 123`).
    //
    // NOTE(eddyb) this also includes the cases when the LHS is also constant
    // (i.e. `reduce` handling constant folding that should be in SPIR-T itself).
    IntBinOpConstRhs(scalar::IntBinOp, Const),

    /// Boolean binary operation with a constant RHS (e.g. `x & true`).
    //
    // NOTE(eddyb) this also includes the cases when the LHS is also constant
    // (i.e. `reduce` handling constant folding that should be in SPIR-T itself).
    BoolBinOpConstRhs(scalar::BoolBinOp, Const),

    BitCast,
    // FIXME(eddyb) include all of `vector::Op` (or obsolete with `flow`).
    VectorExtract {
        elem_idx: u8,
    },

    /// Maps `0` to `false`, and `1` to `true`, but any other input values do
    /// not allow reduction, which is used to signal that the input couldn't
    /// be constrained to `0..=1` (and may take other values).
    //
    // HACK(eddyb) not a real operation, but a helper used to extract a `bool`
    // equivalent for an `OpSwitch`'s scrutinee.
    // FIXME(eddyb) proper SPIR-T range analysis should be implemented and such
    // a reduction not attempted at all if the range is larger than `0..=1`
    // (also, the actual operation can be replaced with `x == 1` or `x != 0`)
    IntToBool,

    // HACK(eddyb) these should be supported by `qptr` itself.
    // FIXME(eddyb) for now these are rewritten into bitcasts and not matched on.
    IntToPtr,
    PtrToInt,
    PtrEqAddr {
        addr: Const,
    },
}

impl TryFrom<(&DataInstKind, &[Option<Const>])> for PureOp {
    type Error = ();
    fn try_from((kind, const_inputs): (&DataInstKind, &[Option<Const>])) -> Result<Self, ()> {
        match kind {
            &DataInstKind::Scalar(scalar::Op::IntUnary(op)) => Ok(Self::IntUnOp(op)),
            &DataInstKind::Scalar(scalar::Op::IntBinary(op)) => {
                match op {
                    // FIXME(eddyb) these produce two outputs each.
                    scalar::IntBinOp::CarryingAdd
                    | scalar::IntBinOp::BorrowingSub
                    | scalar::IntBinOp::WideningMulU
                    | scalar::IntBinOp::WideningMulS => Err(()),

                    _ => Ok(Self::IntBinOpConstRhs(op, const_inputs[1].ok_or(())?)),
                }
            }
            &DataInstKind::Scalar(scalar::Op::BoolUnary(op)) => Ok(Self::BoolUnOp(op)),
            &DataInstKind::Scalar(scalar::Op::BoolBinary(op)) => {
                Ok(Self::BoolBinOpConstRhs(op, const_inputs[1].ok_or(())?))
            }

            &DataInstKind::Vector(vector::Op::Whole(vector::WholeOp::Extract { elem_idx })) => {
                Ok(Self::VectorExtract { elem_idx })
            }
            DataInstKind::SpvInst(spv_inst, lowering) => {
                if lowering.disaggregated_output.is_some()
                    || !lowering.disaggregated_inputs.is_empty()
                {
                    return Err(());
                }

                let wk = &super::SpvSpecWithExtras::get().well_known;

                let op = spv_inst.opcode;
                Ok(match spv_inst.imms[..] {
                    [] if op == wk.OpBitcast => Self::BitCast,

                    [] if op == wk.OpConvertUToPtr => Self::IntToPtr,
                    [] if op == wk.OpConvertPtrToU => Self::PtrToInt,

                    _ => return Err(()),
                })
            }
            _ => Err(()),
        }
    }
}

impl TryFrom<PureOp> for DataInstKind {
    type Error = ();
    fn try_from(op: PureOp) -> Result<Self, ()> {
        let wk = &super::SpvSpecWithExtras::get().well_known;

        let (opcode, imms) = match op {
            PureOp::IntUnOp(op) => return Ok(scalar::Op::from(op).into()),
            PureOp::IntBinOpConstRhs(op, _) => return Ok(scalar::Op::from(op).into()),
            PureOp::BoolUnOp(op) => return Ok(scalar::Op::from(op).into()),
            PureOp::BoolBinOpConstRhs(op, _) => return Ok(scalar::Op::from(op).into()),

            PureOp::BitCast => (wk.OpBitcast, iter::empty().collect()),
            PureOp::VectorExtract { elem_idx } => {
                return Ok(vector::Op::from(vector::WholeOp::Extract { elem_idx }).into());
            }

            PureOp::IntToPtr => (wk.OpConvertUToPtr, iter::empty().collect()),
            PureOp::PtrToInt => (wk.OpConvertPtrToU, iter::empty().collect()),

            // HACK(eddyb) this is the only reason this is `TryFrom` not `From`.
            PureOp::IntToBool | PureOp::PtrEqAddr { .. } => {
                return Err(());
            }
        };
        Ok(DataInstKind::SpvInst(
            spv::Inst { opcode, imms },
            spv::InstLowering::default(),
        ))
    }
}

/// Potentially-reducible application of a `PureOp` (`op`) to `input`.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct Reducible<V = Value> {
    op: PureOp,
    output_type: Type,
    input: V,
}

impl<V> Reducible<V> {
    fn with_input<V2>(self, new_input: V2) -> Reducible<V2> {
        Reducible {
            op: self.op,
            output_type: self.output_type,
            input: new_input,
        }
    }
}

// FIXME(eddyb) instead of taking a `&Context`, could `Reducible` hold a `DataInstForm`?
impl TryFrom<(&Context, &DataInstDef)> for Reducible {
    type Error = ();
    fn try_from((cx, inst_def): (&Context, &DataInstDef)) -> Result<Self, ()> {
        let const_inputs = match inst_def.inputs[..] {
            [x] => [Some(x), None],
            [a, b] => [Some(a), Some(b)],
            _ => return Err(()),
        }
        .map(|v| match v? {
            Value::Const(ct) => Some(ct),
            _ => None,
        });
        let const_inputs = &const_inputs[..inst_def.inputs.len()];

        let inst_form_def = &cx[inst_def.form];
        let op = PureOp::try_from((&inst_form_def.kind, const_inputs))?;

        assert_eq!(inst_form_def.output_types.len(), 1);
        let output_type = inst_form_def.output_types[0];

        match (op, &inst_def.inputs[..]) {
            (PureOp::IntBinOpConstRhs(..) | PureOp::BoolBinOpConstRhs(..), &[input, _])
            | (_, &[input]) => Ok(Self {
                op,
                output_type,
                input,
            }),
            _ => Err(()),
        }
    }
}

impl Reducible {
    // HACK(eddyb) `IntToBool` is the only reason this can return `None`.
    fn try_into_inst(self, cx: &Context) -> Option<DataInstDef> {
        let Self {
            op,
            output_type,
            input,
        } = self;
        Some(DataInstDef {
            attrs: Default::default(),
            form: cx.intern(DataInstFormDef {
                kind: op.try_into().ok()?,
                output_types: [output_type].into_iter().collect(),
            }),
            inputs: match op {
                PureOp::IntBinOpConstRhs(_, rhs) | PureOp::BoolBinOpConstRhs(_, rhs) => {
                    [input, Value::Const(rhs)].into_iter().collect()
                }
                _ => [input].into_iter().collect(),
            },
        })
    }
}

impl Reducible<Const> {
    // FIXME(eddyb) in theory this should always return `Some`.
    fn try_reduce_const(&self, cx: &Context) -> Option<Const> {
        let wk = &super::SpvSpecWithExtras::get().well_known;

        // HACK(eddyb) precomputed here for easy reuse (does not allocate).
        let const_bitcast_spv_inst = spv::Inst {
            opcode: wk.OpSpecConstantOp,
            imms: [spv::Imm::Short(
                wk.LiteralSpecConstantOpInteger,
                wk.OpBitcast.as_u16().into(),
            )]
            .into_iter()
            .collect(),
        };

        let scalar_output_type = self.output_type.as_scalar(cx);
        let scalar_output_from_bits_trunc = |bits: u128| {
            let width = scalar_output_type?.bit_width();
            if width > 128 {
                return None;
            }

            Some(cx.intern(ConstDef {
                attrs: Default::default(),
                ty: self.output_type,
                kind: ConstKind::Scalar(scalar::Const::from_bits(
                    scalar_output_type.unwrap(),
                    bits & (!0u128 >> (128 - width)),
                )),
            }))
        };

        let to_signed = |x: &scalar::Const| match x.ty() {
            scalar::Type::SInt(w) | scalar::Type::UInt(w) => {
                scalar::Const::from_bits(scalar::Type::SInt(w), x.bits())
            }
            _ => *x,
        };

        // FIXME(eddyb) should this just be a method on `scalar::Const`?
        let as_bool = |x: &scalar::Const| match *x {
            scalar::Const::FALSE => Some(false),
            scalar::Const::TRUE => Some(true),
            _ => None,
        };

        let ct_def = &cx[self.input];
        match (self.op, &ct_def.kind) {
            (_, ConstKind::Undef) => Some(cx.intern(ConstDef {
                attrs: ct_def.attrs,
                ty: self.output_type,
                kind: ct_def.kind.clone(),
            })),

            // FIXME(eddyb) these should be in SPIR-T itself.
            (PureOp::IntUnOp(op), ConstKind::Scalar(ct)) => {
                let (x, x_s) = (ct.bits(), to_signed(ct).int_as_i128()?);
                let output_bits = match op {
                    scalar::IntUnOp::Neg => x_s.wrapping_neg() as u128,
                    scalar::IntUnOp::Not => !x,
                    scalar::IntUnOp::CountOnes => x.count_ones().into(),
                    scalar::IntUnOp::TruncOrZeroExtend => x,
                    scalar::IntUnOp::TruncOrSignExtend => x_s as u128,
                };
                scalar_output_from_bits_trunc(output_bits)
            }
            (PureOp::IntBinOpConstRhs(op, ct_rhs), ConstKind::Scalar(ct_lhs)) => {
                let ct_rhs = ct_rhs.as_scalar(cx)?;
                let (a, a_s) = (ct_lhs.bits(), to_signed(ct_lhs).int_as_i128()?);
                let (b, b_s) = (ct_rhs.bits(), to_signed(ct_rhs).int_as_i128()?);
                let output_bits = match op {
                    scalar::IntBinOp::Add => a.wrapping_add(b),
                    scalar::IntBinOp::Sub => a.wrapping_sub(b),
                    scalar::IntBinOp::Mul => a.wrapping_mul(b),
                    scalar::IntBinOp::DivU => a.checked_div(b)?,
                    scalar::IntBinOp::DivS => a_s.checked_div(b_s)? as u128,
                    scalar::IntBinOp::ModU => a.checked_rem(b)?,
                    scalar::IntBinOp::RemS => a_s.checked_rem(b_s)? as u128,
                    scalar::IntBinOp::ModS => return None,
                    scalar::IntBinOp::ShrU => a.checked_shr(b.try_into().ok()?)?,
                    scalar::IntBinOp::ShrS => a_s.checked_shr(b.try_into().ok()?)? as u128,
                    scalar::IntBinOp::Shl => a.checked_shl(b.try_into().ok()?)?,
                    scalar::IntBinOp::Or => a | b,
                    scalar::IntBinOp::Xor => a ^ b,
                    scalar::IntBinOp::And => a & b,
                    scalar::IntBinOp::CarryingAdd
                    | scalar::IntBinOp::BorrowingSub
                    | scalar::IntBinOp::WideningMulU
                    | scalar::IntBinOp::WideningMulS => unreachable!(),
                    scalar::IntBinOp::Eq => (a == b) as u128,
                    scalar::IntBinOp::Ne => (a != b) as u128,
                    scalar::IntBinOp::GtU => (a > b) as u128,
                    scalar::IntBinOp::GtS => (a_s > b_s) as u128,
                    scalar::IntBinOp::GeU => (a >= b) as u128,
                    scalar::IntBinOp::GeS => (a_s >= b_s) as u128,
                    scalar::IntBinOp::LtU => (a < b) as u128,
                    scalar::IntBinOp::LtS => (a_s < b_s) as u128,
                    scalar::IntBinOp::LeU => (a <= b) as u128,
                    scalar::IntBinOp::LeS => (a_s <= b_s) as u128,
                };
                scalar_output_from_bits_trunc(output_bits)
            }
            (PureOp::BoolUnOp(op), ConstKind::Scalar(ct)) => {
                let x = as_bool(ct)?;
                let output = match op {
                    scalar::BoolUnOp::Not => !x,
                };
                Some(cx.intern(scalar::Const::from_bool(output)))
            }
            (PureOp::BoolBinOpConstRhs(op, ct_rhs), ConstKind::Scalar(ct_lhs)) => {
                let ct_rhs = ct_rhs.as_scalar(cx)?;

                let (a, b) = (as_bool(ct_lhs)?, as_bool(ct_rhs)?);
                let output = match op {
                    scalar::BoolBinOp::Eq => a == b,
                    scalar::BoolBinOp::Ne => a != b,
                    scalar::BoolBinOp::Or => a | b,
                    scalar::BoolBinOp::And => a & b,
                };
                Some(cx.intern(scalar::Const::from_bool(output)))
            }

            // FIXME(eddyb) consistently represent all SPIR-T constants with a
            // known bit-pattern.
            (PureOp::BitCast, _) if ct_def.ty == self.output_type => Some(self.input),
            (
                PureOp::BitCast,
                ConstKind::SpvInst {
                    spv_inst_and_const_inputs,
                },
            ) => {
                let (spv_inst, const_inputs) = &**spv_inst_and_const_inputs;
                if *spv_inst == const_bitcast_spv_inst {
                    assert_eq!(const_inputs.len(), 1);
                    self.with_input(const_inputs[0]).try_reduce_const(cx)
                } else if spv_inst.opcode == wk.OpConstantNull {
                    Some(cx.intern(ConstDef {
                        attrs: ct_def.attrs,
                        ty: self.output_type,
                        kind: ConstKind::Scalar(scalar::Const::from_bits(scalar_output_type?, 0)),
                    }))
                } else {
                    None
                }
            }
            (PureOp::BitCast, ConstKind::Scalar(ct)) if scalar_output_type.is_some() => {
                if ct.ty().bit_width() == scalar_output_type?.bit_width() {
                    Some(cx.intern(ConstDef {
                        attrs: ct_def.attrs,
                        ty: self.output_type,
                        kind: ConstKind::Scalar(scalar::Const::try_from_bits(
                            scalar_output_type.unwrap(),
                            ct.bits(),
                        )?),
                    }))
                } else {
                    None
                }
            }
            (PureOp::BitCast, _) => {
                let (spv_inst, const_inputs) =
                    if self.input.as_scalar(cx).map(|ct| ct.bits()) == Some(0) {
                        (wk.OpConstantNull.into(), [].into_iter().collect())
                    } else {
                        (const_bitcast_spv_inst, [self.input].into_iter().collect())
                    };
                Some(cx.intern(ConstDef {
                    attrs: Default::default(),
                    ty: self.output_type,
                    kind: ConstKind::SpvInst {
                        spv_inst_and_const_inputs: Rc::new((spv_inst, const_inputs)),
                    },
                }))
            }

            (PureOp::VectorExtract { elem_idx }, ConstKind::Vector(ct)) => {
                Some(cx.intern(ct.get_elem(elem_idx.into())?))
            }

            (PureOp::IntToBool, ConstKind::Scalar(ct)) => {
                Some(cx.intern(scalar::Const::try_from_bits(scalar::Type::Bool, ct.bits())?))
            }
            (PureOp::PtrEqAddr { .. }, ConstKind::PtrToGlobalVar(_)) => {
                Some(cx.intern(scalar::Const::FALSE))
            }
            (PureOp::PtrEqAddr { addr }, _) => {
                // HACK(eddyb) this effectively expands `PtrEqAddr`'s two-step
                // definition (maybe this is now unnecessary and `Eq` suffices?).
                Self {
                    op: PureOp::IntBinOpConstRhs(scalar::IntBinOp::Eq, addr),
                    input: Self {
                        op: PureOp::BitCast,
                        input: self.input,
                        output_type: cx[addr].ty,
                    }
                    .try_reduce_const(cx)?,
                    ..*self
                }
                .try_reduce_const(cx)
            }

            _ => None,
        }
    }
}

/// Outcome of a single step of a reduction (which may require more steps).
enum ReductionStep {
    Complete(Value),
    Partial(Reducible),
}

impl Reducible {
    // FIXME(eddyb) find better name? (this only inspects constant inputs, and
    // otherwise either returns a constant, or depends on the input uniformly)
    fn try_reduce_shallow(
        &self,
        cx: &Context,
        type_of_value: impl Fn(Value) -> Type,
    ) -> Option<ReductionStep> {
        // HACK(eddyb) aggressive typed rewriting, bypassing even `try_reduce_const`.
        match self.op {
            PureOp::BitCast if type_of_value(self.input) == self.output_type => {
                return Some(ReductionStep::Complete(self.input));
            }

            // HACK(eddyb) rewriting lossless ptr<->int conversions into bitcasts
            // encodes that losslessness and allows simpler reductions downstream.
            PureOp::IntToPtr
                if type_of_value(self.input).as_scalar(cx) == Some(super::QPTR_SIZED_UINT) =>
            {
                return Some(ReductionStep::Partial(Self {
                    op: PureOp::BitCast,
                    ..*self
                }));
            }
            PureOp::PtrToInt if self.output_type.as_scalar(cx) == Some(super::QPTR_SIZED_UINT) => {
                return Some(ReductionStep::Partial(Self {
                    op: PureOp::BitCast,
                    ..*self
                }));
            }

            _ => {}
        }

        if let Value::Const(ct) = self.input {
            return Some(ReductionStep::Complete(Value::Const(
                self.with_input(ct).try_reduce_const(cx)?,
            )));
        }

        // FIXME(eddyb) should this just be a method on `scalar::Const`?
        let as_bool = |x: &scalar::Const| match *x {
            scalar::Const::FALSE => Some(false),
            scalar::Const::TRUE => Some(true),
            _ => None,
        };

        match self.op {
            PureOp::IntBinOpConstRhs(op, ct_rhs) => {
                let ct_rhs = ct_rhs.as_scalar(cx)?;
                match (op, ct_rhs.bits()) {
                    (
                        scalar::IntBinOp::Add
                        | scalar::IntBinOp::Sub
                        | scalar::IntBinOp::ShrU
                        | scalar::IntBinOp::ShrS
                        | scalar::IntBinOp::Shl
                        | scalar::IntBinOp::Or
                        | scalar::IntBinOp::Xor,
                        0,
                    )
                    | (
                        scalar::IntBinOp::Mul | scalar::IntBinOp::DivU | scalar::IntBinOp::DivS,
                        1,
                    ) => Some(ReductionStep::Complete(self.input)),

                    (scalar::IntBinOp::And, mask)
                        if mask == (!0 >> (128 - ct_rhs.ty().bit_width())) =>
                    {
                        Some(ReductionStep::Complete(self.input))
                    }

                    (scalar::IntBinOp::GeU, 0) => Some(ReductionStep::Complete(Value::Const(
                        cx.intern(scalar::Const::TRUE),
                    ))),
                    (scalar::IntBinOp::LtU, 0) => Some(ReductionStep::Complete(Value::Const(
                        cx.intern(scalar::Const::FALSE),
                    ))),

                    _ => None,
                }
            }

            PureOp::BoolBinOpConstRhs(op, ct_rhs) => {
                Some(match (op, as_bool(ct_rhs.as_scalar(cx)?)?) {
                    (scalar::BoolBinOp::Eq, false) | (scalar::BoolBinOp::Ne, true) => {
                        ReductionStep::Partial(Reducible {
                            op: PureOp::BoolUnOp(scalar::BoolUnOp::Not),
                            ..*self
                        })
                    }

                    (scalar::BoolBinOp::Eq | scalar::BoolBinOp::And, true)
                    | (scalar::BoolBinOp::Ne | scalar::BoolBinOp::Or, false) => {
                        ReductionStep::Complete(self.input)
                    }

                    (scalar::BoolBinOp::Or, true) | (scalar::BoolBinOp::And, false) => {
                        ReductionStep::Complete(Value::Const(ct_rhs))
                    }
                })
            }

            _ => None,
        }
    }
}

impl Reducible<&DataInstDef> {
    // FIXME(eddyb) force the input to actually be itself some kind of pure op.
    fn try_reduce_output_of_data_inst(
        &self,
        cx: &Context,
        type_of_value: impl Fn(Value) -> Type,
        output_idx: u32,
    ) -> Option<ReductionStep> {
        // HACK(eddyb) semi-convienient matchable `Reducible | DataInstDef`
        let input_reducible_or_inst_def = {
            let input_inst_def = self.input;
            Reducible::try_from((cx, input_inst_def))
                .ok()
                .ok_or_else(|| {
                    let input_inst_form_def = &cx[input_inst_def.form];
                    (
                        &input_inst_form_def.kind,
                        &input_inst_def.inputs[..],
                        &input_inst_form_def.output_types[..],
                    )
                })
        };

        if input_reducible_or_inst_def.is_ok() {
            assert_eq!(output_idx, 0);
        }

        // NOTE(eddyb) do not destroy information left in e.g. comments.
        #[allow(clippy::match_same_arms)]
        match (self.op, input_reducible_or_inst_def) {
            // HACK(eddyb) workaround for `ptr.is_null()` using `ptr as usize == 0`,
            // and also `enum`s overlapping pointers with unrelated integers.
            (
                PureOp::IntBinOpConstRhs(scalar::IntBinOp::Eq, ct_addr),
                Ok(Reducible {
                    op: PureOp::BitCast,
                    input: ptr_input,
                    ..
                }),
            ) if matches!(cx[type_of_value(ptr_input)].kind, TypeKind::QPtr) => {
                return Some(ReductionStep::Partial(Reducible {
                    op: PureOp::PtrEqAddr { addr: ct_addr },
                    output_type: self.output_type,
                    input: ptr_input,
                }));
            }

            (
                PureOp::BoolUnOp(scalar::BoolUnOp::Not),
                Ok(Reducible {
                    op: PureOp::BoolUnOp(scalar::BoolUnOp::Not),
                    input,
                    ..
                }),
            ) => {
                return Some(ReductionStep::Complete(input));
            }

            (
                PureOp::IntUnOp(_)
                | PureOp::IntBinOpConstRhs(..)
                | PureOp::BoolUnOp(_)
                | PureOp::BoolBinOpConstRhs(..),
                _,
            ) => {
                // FIXME(eddyb) reduce compositions.
            }

            (
                PureOp::BitCast,
                Ok(Reducible {
                    op: PureOp::BitCast,
                    input,
                    ..
                }),
            ) => {
                return Some(ReductionStep::Partial(self.with_input(input)));
            }
            (PureOp::BitCast, _) => {}

            (
                PureOp::VectorExtract {
                    elem_idx: extract_idx,
                },
                Err((
                    &DataInstKind::Vector(vector::Op::Whole(vector::WholeOp::Insert {
                        elem_idx: insert_idx,
                    })),
                    &[new_elem, prev_vector],
                    _,
                )),
            ) => {
                return Some(if insert_idx == extract_idx {
                    ReductionStep::Complete(new_elem)
                } else {
                    ReductionStep::Partial(self.with_input(prev_vector))
                });
            }
            (PureOp::VectorExtract { .. }, _) => {}

            (PureOp::IntToBool, _) => {
                // FIXME(eddyb) look into what instructions might end up
                // being used to transform booleans into integers.
            }

            (PureOp::IntToPtr, _) => {}
            (PureOp::PtrToInt, _) => {}
            (
                PureOp::PtrEqAddr { .. },
                Err((
                    DataInstKind::QPtr(
                        // FIXME(eddyb) some of these might need an "inbounds" flag.
                        spirt::qptr::QPtrOp::FuncLocalVar(_)
                        | spirt::qptr::QPtrOp::HandleArrayIndex
                        | spirt::qptr::QPtrOp::BufferData
                        | spirt::qptr::QPtrOp::Offset(_)
                        | spirt::qptr::QPtrOp::DynOffset { .. },
                    ),
                    ..,
                )),
            ) => {
                return Some(ReductionStep::Complete(Value::Const(
                    cx.intern(scalar::Const::FALSE),
                )));
            }
            (
                PureOp::PtrEqAddr { addr },
                Ok(Reducible {
                    op: PureOp::BitCast,
                    input,
                    ..
                }),
            ) if type_of_value(input) == cx[addr].ty => {
                return Some(ReductionStep::Partial(Reducible {
                    op: PureOp::IntBinOpConstRhs(scalar::IntBinOp::Eq, addr),
                    output_type: self.output_type,
                    input,
                }));
            }
            (PureOp::PtrEqAddr { .. }, _) => {}
        }

        None
    }
}

// FIXME(eddyb) `Result<Value, Incomplete<Reducible>>` is isomorphic to the
// existing `ReductionStep`, only reason to do it this way is the difference
// between a single step and a composition of steps, but that's not great.
#[derive(Copy, Clone)]
struct Incomplete<T>(T);

impl Reducible {
    // FIXME(eddyb) make this into some kind of local `ReduceCx` method.
    fn try_reduce(
        self,
        cx: &Context,
        // FIXME(eddyb) come up with a better convention for this!
        func: FuncAtMut<'_, ()>,

        parent_map: &ParentMap,

        cache: &mut FxHashMap<Self, Option<Result<Value, Incomplete<Self>>>>,
    ) -> Option<Value> {
        self.try_reduce_to_value_or_incomplete(cx, func, parent_map, cache)?
            .ok()
    }

    // FIXME(eddyb) make this into some kind of local `ReduceCx` method.
    fn try_reduce_to_value_or_incomplete(
        self,
        cx: &Context,
        // FIXME(eddyb) come up with a better convention for this!
        mut func: FuncAtMut<'_, ()>,

        parent_map: &ParentMap,

        cache: &mut FxHashMap<Self, Option<Result<Value, Incomplete<Self>>>>,
    ) -> Option<Result<Value, Incomplete<Self>>> {
        if let Some(&cached) = cache.get(&self) {
            return cached;
        }

        let result = match self.try_reduce_uncached_step(cx, func.reborrow(), parent_map, cache) {
            // FIXME(eddyb) actually use a loop instead of recursing here,
            // but that can't easily handle caching every single step.
            Some(ReductionStep::Partial(redu)) => Some(
                redu.try_reduce_to_value_or_incomplete(cx, func, parent_map, cache)
                    .unwrap_or(Err(Incomplete(redu))),
            ),
            Some(ReductionStep::Complete(v)) => Some(Ok(v)),
            None => None,
        };

        cache.insert(self, result);

        result
    }

    // FIXME(eddyb) make this into some kind of local `ReduceCx` method.
    fn try_reduce_uncached_step(
        self,
        cx: &Context,
        // FIXME(eddyb) come up with a better convention for this!
        mut func: FuncAtMut<'_, ()>,

        parent_map: &ParentMap,

        cache: &mut FxHashMap<Self, Option<Result<Value, Incomplete<Self>>>>,
    ) -> Option<ReductionStep> {
        let as_const = |v: Value| match v {
            Value::Const(ct) => Some(ct),
            _ => None,
        };

        {
            let func = func.reborrow().freeze();
            if let Some(step) = self.try_reduce_shallow(cx, |v| func.at(v).type_of(cx)) {
                return Some(step);
            }
        }

        match self.input {
            Value::Const(_) => None,
            Value::ControlRegionInput {
                region,
                input_idx: state_idx,
            } => {
                // HACK(eddyb) avoid generating `bool.not` instructions in loops,
                // which could lead to endless cycling (see `FlipIfElseCond`).
                if let PureOp::BoolUnOp(scalar::BoolUnOp::Not) = self.op {
                    return None;
                }

                let loop_node = *parent_map.control_region_parent.get(region)?;
                // HACK(eddyb) this can't be a closure due to lifetime elision.
                fn loop_initial_states(
                    func_at_loop_node: FuncAtMut<'_, ControlNode>,
                ) -> &mut SmallVec<[Value; 2]> {
                    match &mut func_at_loop_node.def().kind {
                        ControlNodeKind::Loop { initial_inputs, .. } => initial_inputs,
                        _ => unreachable!(),
                    }
                }

                let input_from_initial_state =
                    loop_initial_states(func.reborrow().at(loop_node))[state_idx as usize];
                let input_from_updated_state =
                    func.reborrow().at(region).def().outputs[state_idx as usize];

                let output_from_initial_state = self
                    .with_input(input_from_initial_state)
                    .try_reduce(cx, func.reborrow(), parent_map, cache)?;
                // HACK(eddyb) this is here because it can fail, see the comment
                // on `output_from_updated_state` for what's actually going on.
                let output_from_updated_state_inst = self
                    .with_input(input_from_updated_state)
                    .try_into_inst(cx)?;

                // Now that the reduction succeeded for the initial state,
                // we can proceed with augmenting the loop with the extra state.
                loop_initial_states(func.reborrow().at(loop_node)).push(output_from_initial_state);

                let loop_state_decls = &mut func.reborrow().at(region).def().inputs;
                let new_loop_state_idx = u32::try_from(loop_state_decls.len()).unwrap();
                loop_state_decls.push(ControlRegionInputDecl {
                    attrs: Default::default(),
                    ty: self.output_type,
                });

                // HACK(eddyb) generating the instruction wholesale again is not
                // the most efficient way to go about this, but avoiding getting
                // stuck in a loop while processing a loop is also important.
                //
                // FIXME(eddyb) attempt to replace this with early-inserting in
                // `cache` *then* returning.
                let output_from_updated_state = func
                    .data_insts
                    .define(cx, output_from_updated_state_inst.into());
                func.reborrow()
                    .at(region)
                    .def()
                    .outputs
                    .push(Value::DataInstOutput {
                        inst: output_from_updated_state,
                        output_idx: 0,
                    });

                // FIXME(eddyb) move this into some kind of utility/common helpers.
                let loop_body_last_block = func
                    .reborrow()
                    .at(region)
                    .def()
                    .children
                    .iter()
                    .last
                    .filter(|&node| {
                        matches!(
                            func.reborrow().at(node).def().kind,
                            ControlNodeKind::Block { .. }
                        )
                    })
                    .unwrap_or_else(|| {
                        let new_block = func.control_nodes.define(
                            cx,
                            ControlNodeDef {
                                kind: ControlNodeKind::Block {
                                    insts: Default::default(),
                                },
                                outputs: Default::default(),
                            }
                            .into(),
                        );
                        func.control_regions[region]
                            .children
                            .insert_last(new_block, func.control_nodes);
                        new_block
                    });
                match &mut func.control_nodes[loop_body_last_block].kind {
                    ControlNodeKind::Block { insts } => {
                        insts.insert_last(output_from_updated_state, func.data_insts);
                    }
                    _ => unreachable!(),
                }

                Some(ReductionStep::Complete(Value::ControlRegionInput {
                    region,
                    input_idx: new_loop_state_idx,
                }))
            }
            Value::ControlNodeOutput {
                control_node,
                output_idx,
            } => {
                let cases = match &func.reborrow().at(control_node).def().kind {
                    ControlNodeKind::Select { cases, .. } => cases,
                    // NOTE(eddyb) only `Select`s can have outputs right now.
                    _ => unreachable!(),
                };

                // FIXME(eddyb) remove all the cloning and undo additions of new
                // outputs "upstream", if they end up unused (or let DCE do it?).
                let cases = cases.clone();
                let per_case_new_output: SmallVec<[_; 2]> = cases
                    .iter()
                    .map(|&case| {
                        let per_case_input =
                            func.reborrow().at(case).def().outputs[output_idx as usize];
                        self.with_input(per_case_input).try_reduce(
                            cx,
                            func.reborrow(),
                            parent_map,
                            cache,
                        )
                    })
                    .collect::<Option<_>>()?;

                // Try to avoid introducing a new output, by reducing the merge
                // of the per-case output values to a single value, if possible.
                {
                    let func = func.reborrow().freeze();
                    let (kind, scrutinee) = match &func.at(control_node).def().kind {
                        ControlNodeKind::Select {
                            kind, scrutinee, ..
                        } => (kind, *scrutinee),
                        _ => unreachable!(),
                    };
                    if let Some(v) = try_reduce_select(
                        cx,
                        parent_map,
                        func,
                        Some(control_node),
                        kind,
                        scrutinee,
                        per_case_new_output.iter().copied(),
                    ) {
                        return Some(ReductionStep::Complete(v));
                    }

                    // HACK(eddyb) avoid adding boolean constants to the cases of
                    // any `if`-`else`s just to negate the original condition.
                    if let (SelectionKind::BoolCond, [t, e]) = (kind, &per_case_new_output[..]) {
                        if [t, e].map(|&x| as_const(x)?.as_scalar(cx))
                            == [Some(&scalar::Const::FALSE), Some(&scalar::Const::TRUE)]
                        {
                            return Some(ReductionStep::Partial(Reducible {
                                op: PureOp::BoolUnOp(scalar::BoolUnOp::Not),
                                output_type: self.output_type,
                                input: scrutinee,
                            }));
                        }
                    }
                }

                // HACK(eddyb) avoid generating e.g. `if ... { false } else { true }`,
                // which could lead to endless cycling (see `FlipIfElseCond`).
                if let PureOp::BoolUnOp(scalar::BoolUnOp::Not) = self.op {
                    return None;
                }

                // Merge the per-case output values into a new output.
                let control_node_output_decls = &mut func.reborrow().at(control_node).def().outputs;
                let new_output_idx = u32::try_from(control_node_output_decls.len()).unwrap();
                control_node_output_decls.push(ControlNodeOutputDecl {
                    attrs: Default::default(),
                    ty: self.output_type,
                });
                for (&case, new_output) in cases.iter().zip(per_case_new_output) {
                    let per_case_outputs = &mut func.reborrow().at(case).def().outputs;
                    assert_eq!(per_case_outputs.len(), new_output_idx as usize);
                    per_case_outputs.push(new_output);
                }
                Some(ReductionStep::Complete(Value::ControlNodeOutput {
                    control_node,
                    output_idx: new_output_idx,
                }))
            }
            Value::DataInstOutput { inst, output_idx } => {
                // HACK(eddyb) special-casing `OpSelect` like this shouldn't even
                // be needed, it should be replaced with `Select` control nodes.
                let inst_def = &*func.reborrow().at(inst).def();
                if let DataInstKind::SpvInst(spv_inst, lowering) = &cx[inst_def.form].kind {
                    let wk = &super::SpvSpecWithExtras::get().well_known;
                    if spv_inst.opcode == wk.OpSelect
                        && lowering.disaggregated_output.is_none()
                        && lowering.disaggregated_inputs.is_empty()
                    {
                        let select_cond = inst_def.inputs[0];
                        let per_case_inputs = [inst_def.inputs[1], inst_def.inputs[2]];
                        let per_case_new_output = per_case_inputs.map(|per_case_input| {
                            self.with_input(per_case_input).try_reduce(
                                cx,
                                func.reborrow(),
                                parent_map,
                                cache,
                            )
                        });
                        if let [Some(t), Some(e)] = per_case_new_output {
                            if let Some(new_output) = try_reduce_select(
                                cx,
                                parent_map,
                                func.reborrow().freeze(),
                                None,
                                &SelectionKind::BoolCond,
                                select_cond,
                                [t, e].into_iter(),
                            ) {
                                return Some(ReductionStep::Complete(new_output));
                            }

                            // FIXME(eddyb) this should be part of `try_reduce_select`.
                            if [t, e].map(|x| as_const(x)?.as_scalar(cx))
                                == [Some(&scalar::Const::FALSE), Some(&scalar::Const::TRUE)]
                            {
                                return Some(ReductionStep::Partial(Reducible {
                                    op: PureOp::BoolUnOp(scalar::BoolUnOp::Not),
                                    output_type: self.output_type,
                                    input: select_cond,
                                }));
                            }
                        }
                    }
                }

                let func = func.reborrow().freeze();
                self.with_input(func.at(inst).def())
                    .try_reduce_output_of_data_inst(cx, |v| func.at(v).type_of(cx), output_idx)
            }
        }
    }
}
