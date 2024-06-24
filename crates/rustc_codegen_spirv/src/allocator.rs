// FIXME(eddyb) this should be implemented once for all backends, it doesn't
// need backend-specific logic, just a way to create modules and functions.

use rspirv::spirv::{FunctionControl, LinkageType, StorageClass, Word};
use rustc_ast::expand::allocator::{
    alloc_error_handler_name, default_fn_name, global_fn_name, AllocatorKind, AllocatorTy,
    ALLOCATOR_METHODS, NO_ALLOC_SHIM_IS_UNSTABLE,
};
use rustc_codegen_ssa::traits::{
    AbiBuilderMethods as _, BaseTypeMethods as _, BuilderMethods as _, ConstMethods as _,
};
use rustc_session::config::OomStrategy;
use rustc_span::DUMMY_SP;

use crate::builder::Builder;
use crate::builder_spirv::{SpirvValue, SpirvValueExt as _};
use crate::codegen_cx::CodegenCx;
use crate::spirv_type::SpirvType;

pub(crate) fn codegen(
    cx: &CodegenCx<'_>,
    kind: AllocatorKind,
    alloc_error_handler_kind: AllocatorKind,
) {
    let usize = cx.type_usize();
    let i8 = cx.type_i8();
    let i8p = cx.type_ptr_to(i8);

    if kind == AllocatorKind::Default {
        for method in ALLOCATOR_METHODS {
            let mut args = Vec::with_capacity(method.inputs.len());
            for input in method.inputs.iter() {
                match input.ty {
                    AllocatorTy::Layout => {
                        args.push(usize); // size
                        args.push(usize); // align
                    }
                    AllocatorTy::Ptr => args.push(i8p),
                    AllocatorTy::Usize => args.push(usize),

                    AllocatorTy::ResultPtr | AllocatorTy::Unit => panic!("invalid allocator arg"),
                }
            }
            let output = match method.output {
                AllocatorTy::ResultPtr => Some(i8p),
                AllocatorTy::Unit => None,

                AllocatorTy::Layout | AllocatorTy::Usize | AllocatorTy::Ptr => {
                    panic!("invalid allocator output")
                }
            };

            let from_name = global_fn_name(method.name);
            let to_name = default_fn_name(method.name);

            create_wrapper_function(cx, &from_name, &to_name, &args, output, false);
        }
    }

    // rust alloc error handler
    create_wrapper_function(
        cx,
        "__rust_alloc_error_handler",
        alloc_error_handler_name(alloc_error_handler_kind),
        &[usize, usize], // size, align
        None,
        true,
    );

    let define_global = |name: &str, init: SpirvValue| {
        let init_val_id = init.def_cx(cx);
        let ptr_ty = cx.type_ptr_to(init.ty);
        let global_id =
            cx.emit_global()
                .variable(ptr_ty, None, StorageClass::Private, Some(init_val_id));
        cx.set_linkage(global_id, name.to_string(), LinkageType::Export);
    };

    // __rust_alloc_error_handler_should_panic
    define_global(
        OomStrategy::SYMBOL,
        cx.const_u8(cx.tcx.sess.opts.unstable_opts.oom.should_panic()),
    );

    define_global(NO_ALLOC_SHIM_IS_UNSTABLE, cx.const_u8(0));
}

fn create_wrapper_function(
    cx: &CodegenCx<'_>,
    from_name: &str,
    to_name: &str,
    // NOTE(eddyb) these are SPIR-V type IDs.
    args: &[Word],
    output: Option<Word>,
    _no_return: bool,
) {
    let ret_ty = output.unwrap_or_else(|| SpirvType::Void.def(DUMMY_SP, cx));
    let fn_ty = cx.type_func(args, ret_ty);
    let decl_fn = |name: &str, linkage_type| {
        let mut emit = cx.emit_global();
        let fn_id = emit
            .begin_function(ret_ty, None, FunctionControl::NONE, fn_ty)
            .unwrap();
        let parameter_values = args
            .iter()
            .map(|&ty| emit.function_parameter(ty).unwrap().with_type(ty))
            .collect::<Vec<_>>();
        cx.function_parameter_values
            .borrow_mut()
            .insert(fn_id, parameter_values);
        emit.end_function().unwrap();
        drop(emit);
        cx.set_linkage(fn_id, name.to_string(), linkage_type);
        fn_id.with_type(fn_ty)
    };
    let wrapper = decl_fn(from_name, LinkageType::Export);
    let callee = decl_fn(to_name, LinkageType::Import);

    let mut bx = Builder::build(cx, Builder::append_block(cx, wrapper, ""));
    let call_args = (0..args.len()).map(|i| bx.get_param(i)).collect::<Vec<_>>();
    let ret = bx.call(callee.ty, None, None, callee, &call_args, None);
    if output.is_some() {
        bx.ret(ret);
    } else {
        bx.ret_void();
    }
}
