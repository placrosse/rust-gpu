use crate::things::{SpirvModuleBuffer, SpirvThinBuffer};
use crate::SpirvCodegenBackend;
use rustc_codegen_ssa::back::lto::{LtoModuleCodegen, SerializedModule, ThinModule, ThinShared};
use rustc_codegen_ssa::back::write::CodegenContext;
use rustc_codegen_ssa::CodegenResults;
use rustc_errors::FatalError;
use rustc_middle::dep_graph::WorkProduct;
use rustc_middle::middle::cstore::NativeLib;
use rustc_middle::middle::dependency_format::Linkage;
use rustc_session::config::{CrateType, Lto, OutputFilenames, OutputType};
use rustc_session::output::{check_file_is_writeable, invalid_output_for_target, out_filename};
use rustc_session::utils::NativeLibKind;
use rustc_session::Session;
use std::collections::HashSet;
use std::ffi::CString;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tar::{Archive, Builder};

pub fn link<'a>(
    sess: &'a Session,
    codegen_results: &CodegenResults,
    outputs: &OutputFilenames,
    crate_name: &str,
) {
    let output_metadata = sess.opts.output_types.contains_key(&OutputType::Metadata);
    for &crate_type in sess.crate_types().iter() {
        if (sess.opts.debugging_opts.no_codegen || !sess.opts.output_types.should_codegen())
            && !output_metadata
            && crate_type == CrateType::Executable
        {
            continue;
        }

        if invalid_output_for_target(sess, crate_type) {
            panic!(
                "invalid output type `{:?}` for target os `{}`",
                crate_type, sess.opts.target_triple
            );
        }

        for obj in codegen_results
            .modules
            .iter()
            .filter_map(|m| m.object.as_ref())
        {
            check_file_is_writeable(obj, sess);
        }

        if outputs.outputs.should_codegen() {
            let out_filename = out_filename(sess, crate_type, outputs, crate_name);
            match crate_type {
                CrateType::Rlib => {
                    link_rlib(codegen_results, &out_filename);
                }
                CrateType::Executable | CrateType::Cdylib => {
                    link_exe(sess, crate_type, &out_filename, codegen_results)
                }
                other => panic!("CrateType {:?} not supported yet", other),
            }
        }
    }
}

fn link_rlib(codegen_results: &CodegenResults, out_filename: &Path) {
    let mut file_list = Vec::<&Path>::new();
    for obj in codegen_results
        .modules
        .iter()
        .filter_map(|m| m.object.as_ref())
    {
        file_list.push(obj);
    }
    for lib in codegen_results.crate_info.used_libraries.iter() {
        match lib.kind {
            NativeLibKind::StaticBundle => {}
            NativeLibKind::StaticNoBundle
            | NativeLibKind::Dylib
            | NativeLibKind::Framework
            | NativeLibKind::RawDylib
            | NativeLibKind::Unspecified => continue,
        }
        if let Some(name) = lib.name {
            panic!("Adding native library to rlib not supported yet: {}", name);
        }
    }

    // let metadata = &codegen_results.metadata.raw_data;
    // file_list.push(metadata);

    create_archive(&file_list, out_filename);
}

fn link_exe(
    sess: &Session,
    crate_type: CrateType,
    out_filename: &Path,
    codegen_results: &CodegenResults,
) {
    let mut objects = Vec::new();
    let mut rlibs = Vec::new();
    for obj in codegen_results
        .modules
        .iter()
        .filter_map(|m| m.object.as_ref())
    {
        objects.push(obj.clone());
    }

    link_local_crate_native_libs_and_dependent_crate_libs(
        &mut rlibs,
        sess,
        crate_type,
        codegen_results,
    );

    do_link(&objects, &rlibs, out_filename);
}

fn link_local_crate_native_libs_and_dependent_crate_libs<'a>(
    rlibs: &mut Vec<PathBuf>,
    sess: &'a Session,
    crate_type: CrateType,
    codegen_results: &CodegenResults,
) {
    if sess.opts.debugging_opts.link_native_libraries {
        add_local_native_libraries(sess, codegen_results);
    }
    add_upstream_rust_crates(rlibs, codegen_results, crate_type);
    if sess.opts.debugging_opts.link_native_libraries {
        add_upstream_native_libraries(sess, codegen_results, crate_type);
    }
}

fn add_local_native_libraries(sess: &Session, codegen_results: &CodegenResults) {
    let relevant_libs = codegen_results
        .crate_info
        .used_libraries
        .iter()
        .filter(|l| relevant_lib(sess, l));
    assert_eq!(relevant_libs.count(), 0);
}

fn add_upstream_rust_crates(
    rlibs: &mut Vec<PathBuf>,
    codegen_results: &CodegenResults,
    crate_type: CrateType,
) {
    let (_, data) = codegen_results
        .crate_info
        .dependency_formats
        .iter()
        .find(|(ty, _)| *ty == crate_type)
        .expect("failed to find crate type in dependency format list");
    let deps = &codegen_results.crate_info.used_crates_dynamic;
    for &(cnum, _) in deps.iter() {
        let src = &codegen_results.crate_info.used_crate_source[&cnum];
        match data[cnum.as_usize() - 1] {
            Linkage::NotLinked | Linkage::IncludedFromDylib => {}
            Linkage::Static => rlibs.push(src.rlib.as_ref().unwrap().0.clone()),
            //Linkage::Dynamic => rlibs.push(src.dylib.as_ref().unwrap().0.clone()),
            Linkage::Dynamic => panic!("TODO: Linkage::Dynamic not supported yet"),
        }
    }
}

fn add_upstream_native_libraries(
    sess: &Session,
    codegen_results: &CodegenResults,
    crate_type: CrateType,
) {
    let (_, data) = codegen_results
        .crate_info
        .dependency_formats
        .iter()
        .find(|(ty, _)| *ty == crate_type)
        .expect("failed to find crate type in dependency format list");

    let crates = &codegen_results.crate_info.used_crates_static;
    for &(cnum, _) in crates {
        for lib in codegen_results.crate_info.native_libraries[&cnum].iter() {
            let name = match lib.name {
                Some(l) => l,
                None => continue,
            };
            if !relevant_lib(sess, &lib) {
                continue;
            }
            match lib.kind {
                NativeLibKind::Dylib | NativeLibKind::Unspecified => {
                    panic!("TODO: dylib nativelibkind not supported yet: {}", name)
                }
                NativeLibKind::Framework => {
                    panic!("TODO: framework nativelibkind not supported yet: {}", name)
                }
                NativeLibKind::StaticNoBundle => {
                    if data[cnum.as_usize() - 1] == Linkage::Static {
                        panic!(
                            "TODO: staticnobundle nativelibkind not supported yet: {}",
                            name
                        );
                    }
                }
                NativeLibKind::StaticBundle => {}
                NativeLibKind::RawDylib => {
                    panic!("raw_dylib feature not yet implemented: {}", name);
                }
            }
        }
    }
}

fn relevant_lib(sess: &Session, lib: &NativeLib) -> bool {
    match lib.cfg {
        Some(ref cfg) => rustc_attr::cfg_matches(cfg, &sess.parse_sess, None),
        None => true,
    }
}

fn create_archive(files: &[&Path], out_filename: &Path) {
    let file = File::create(out_filename).unwrap();
    let mut builder = Builder::new(file);
    let mut filenames = HashSet::new();
    for file in files {
        assert!(
            filenames.insert(file.file_name().unwrap()),
            "Duplicate filename in archive: {:?}",
            file.file_name().unwrap()
        );
        builder
            .append_path_with_name(file, file.file_name().unwrap())
            .unwrap();
    }
    builder.into_inner().unwrap();
}

fn do_link(objects: &[PathBuf], rlibs: &[PathBuf], out_filename: &Path) {
    let mut modules = Vec::new();
    for obj in objects {
        let mut bytes = Vec::new();
        File::open(obj).unwrap().read_to_end(&mut bytes).unwrap();
        modules.push(load(&bytes));
    }
    for rlib in rlibs {
        for entry in Archive::new(File::open(rlib).unwrap()).entries().unwrap() {
            let mut bytes = Vec::new();
            entry.unwrap().read_to_end(&mut bytes).unwrap();
            modules.push(load(&bytes));
        }
    }

    let mut module_refs = modules.iter_mut().collect::<Vec<_>>();

    let result = rspirv_linker::link(
        &mut module_refs,
        &rspirv_linker::Options {
            lib: false,
            partial: false,
        },
    )
    .unwrap();

    use rspirv::binary::Assemble;
    File::create(out_filename)
        .unwrap()
        .write_all(crate::slice_u32_to_u8(&result.assemble()))
        .unwrap();

    fn load(bytes: &[u8]) -> rspirv::dr::Module {
        let mut loader = rspirv::dr::Loader::new();
        rspirv::binary::parse_bytes(&bytes, &mut loader).unwrap();
        loader.module()
    }
}

pub(crate) fn run_thin(
    cgcx: &CodegenContext<SpirvCodegenBackend>,
    modules: Vec<(String, SpirvThinBuffer)>,
    cached_modules: Vec<(SerializedModule<SpirvModuleBuffer>, WorkProduct)>,
) -> Result<(Vec<LtoModuleCodegen<SpirvCodegenBackend>>, Vec<WorkProduct>), FatalError> {
    if cgcx.opts.cg.linker_plugin_lto.enabled() {
        unreachable!("We should never reach this case if the LTO step is deferred to the linker");
    }
    if cgcx.lto != Lto::ThinLocal {
        for _ in cgcx.each_linked_rlib_for_lto.iter() {
            panic!("TODO: Implement whatever the heck this is");
        }
    }
    let mut thin_buffers = Vec::with_capacity(modules.len());
    let mut module_names = Vec::with_capacity(modules.len() + cached_modules.len());

    for (name, buffer) in modules.into_iter() {
        let cname = CString::new(name.clone()).unwrap();
        thin_buffers.push(buffer);
        module_names.push(cname);
    }

    let mut serialized_modules = Vec::with_capacity(cached_modules.len());

    for (sm, wp) in cached_modules {
        let _slice_u8 = sm.data();
        serialized_modules.push(sm);
        module_names.push(CString::new(wp.cgu_name).unwrap());
    }

    let shared = Arc::new(ThinShared {
        data: (),
        thin_buffers,
        serialized_modules,
        module_names,
    });

    let mut opt_jobs = vec![];
    for (module_index, _) in shared.module_names.iter().enumerate() {
        opt_jobs.push(LtoModuleCodegen::Thin(ThinModule {
            shared: shared.clone(),
            idx: module_index,
        }));
    }

    Ok((opt_jobs, vec![]))
}