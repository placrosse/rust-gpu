use rustc_data_structures::fx::FxHashMap;
use spirt::func_at::FuncAt;
use spirt::transform::InnerInPlaceTransform;
use spirt::visit::InnerVisit;
use spirt::{
    Context, ControlNode, ControlNodeKind, ControlRegion, FuncDefBody, SelectionKind, Value,
};
use std::mem;

use super::{ReplaceValueWith, VisitAllControlRegionsAndNodes};

/// Combine consecutive `Select`s in `func_def_body`.
pub(crate) fn fuse_selects_in_func(_cx: &Context, func_def_body: &mut FuncDefBody) {
    // Avoid having to support unstructured control-flow.
    if func_def_body.unstructured_cfg.is_some() {
        return;
    }

    // HACK(eddyb) this kind of random-access is easier than using `spirt::transform`.
    let mut all_regions = vec![];

    func_def_body.inner_visit_with(&mut VisitAllControlRegionsAndNodes {
        state: (),
        enter_control_region: |_: &mut (), func_at_control_region: FuncAt<'_, ControlRegion>| {
            all_regions.push(func_at_control_region.position);
        },
        exit_control_region: |_: &mut (), _| {},
        enter_control_node: |_: &mut (), _| {},
        exit_control_node: |_: &mut (), _| {},
    });

    // HACK(eddyb) `fused_control_node -> (base_control_node, fused_outputs_start)`,
    // used to remap the outputs of `Select`s that were fused together.
    let mut rebased_outputs = FxHashMap::default();
    let maybe_rebase = |rebased_outputs: &FxHashMap<ControlNode, (ControlNode, u32)>, v| match v {
        Value::ControlNodeOutput {
            control_node,
            output_idx,
        } => {
            // NOTE(eddyb) this doesn't require a loop because the target of
            // fusion (`base_control_node`) cannot itself be fused in turn.
            let &(base_control_node, fused_outputs_start) = rebased_outputs.get(&control_node)?;
            Some(Value::ControlNodeOutput {
                control_node: base_control_node,
                output_idx: fused_outputs_start + output_idx,
            })
        }
        _ => None,
    };

    for region in all_regions {
        let mut func_at_children_iter = func_def_body.at_mut(region).at_children().into_iter();
        while let Some(func_at_child) = func_at_children_iter.next() {
            let base_control_node = func_at_child.position;
            if let ControlNodeKind::Select {
                kind: SelectionKind::BoolCond,
                scrutinee,
                cases,
            } = &func_at_child.def().kind
            {
                let &base_cond = scrutinee;
                let base_cases = cases.clone();

                // Scan ahead for candidate `Select`s (with the same condition).
                let mut fusion_candidate_iter = func_at_children_iter.reborrow();
                while let Some(func_at_fusion_candidate) = fusion_candidate_iter.next() {
                    let fusion_candidate = func_at_fusion_candidate.position;
                    let mut func = func_at_fusion_candidate.at(());
                    let fusion_candidate_def = func.reborrow().at(fusion_candidate).def();
                    match &mut fusion_candidate_def.kind {
                        // HACK(eddyb) ignore empty blocks (created by
                        // e.g. `remove_unused_values_in_func`).
                        ControlNodeKind::Block { insts } if insts.is_empty() => {}

                        ControlNodeKind::Select {
                            kind: SelectionKind::BoolCond,
                            scrutinee: fusion_candidate_cond,
                            cases: fusion_candidate_cases,
                        } if *fusion_candidate_cond == base_cond => {
                            let cases_to_fuse = mem::take(fusion_candidate_cases);
                            let fusion_candidate_outputs =
                                mem::take(&mut fusion_candidate_def.outputs);

                            // Concatenate the `Select`s' respective cases
                            // ("then" with "then", "else" with "else", etc.).
                            for (&base_case, &case_to_fuse) in base_cases.iter().zip(&cases_to_fuse)
                            {
                                // Replace uses of the outputs of the first `Select`,
                                // in the second one's case, with the specific values
                                // (e.g. `let y = if c { x } ...; if c { f(y) }`
                                // has to become `let y = if c { f(x); x } ...`).
                                //
                                // FIXME(eddyb) avoid cloning here.
                                let outputs_of_base_case =
                                    func.reborrow().at(base_case).def().outputs.clone();
                                func.reborrow()
                                    .at(case_to_fuse)
                                    .inner_in_place_transform_with(&mut ReplaceValueWith(|v| {
                                        match maybe_rebase(&rebased_outputs, v).unwrap_or(v) {
                                            Value::ControlNodeOutput {
                                                control_node,
                                                output_idx,
                                            } if control_node == base_control_node => {
                                                Some(outputs_of_base_case[output_idx as usize])
                                            }

                                            _ => None,
                                        }
                                    }));

                                let case_to_fuse_def =
                                    mem::take(func.reborrow().at(case_to_fuse).def());

                                assert_eq!(case_to_fuse_def.inputs.len(), 0);

                                let base_case_def = &mut func.control_regions[base_case];
                                base_case_def
                                    .children
                                    .append(case_to_fuse_def.children, func.control_nodes);
                                base_case_def.outputs.extend(case_to_fuse_def.outputs);
                            }

                            // HACK(eddyb) because we can't remove list elements yet,
                            // we instead replace the `Select` with an empty `Block`.
                            // FIXME(eddyb) removing is definitely possible now!
                            func.reborrow().at(fusion_candidate).def().kind =
                                ControlNodeKind::Block {
                                    insts: Default::default(),
                                };

                            // Append the second `Select`'s outputs to the first's.
                            if !fusion_candidate_outputs.is_empty() {
                                let base_outputs =
                                    &mut func.reborrow().at(base_control_node).def().outputs;
                                let fused_outputs_start =
                                    u32::try_from(base_outputs.len()).unwrap();
                                rebased_outputs.insert(
                                    fusion_candidate,
                                    (base_control_node, fused_outputs_start),
                                );
                                base_outputs.extend(fusion_candidate_outputs);
                            }
                        }

                        _ => break,
                    }
                }
            }
        }
    }
    func_def_body.inner_in_place_transform_with(&mut ReplaceValueWith(|v| {
        maybe_rebase(&rebased_outputs, v)
    }));
}
