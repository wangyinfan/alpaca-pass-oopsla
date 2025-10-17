#include "include/AlpacaPass.h"

gv_vec gv_list;

/*
 * Collect all global variables whose names contain "_global_".
 * NOTE: In LLVM 6.0, Module::global_iterator no longer implicitly converts
 * to GlobalVariable*, so we must take the address explicitly using &*GI.
 */
void get_gv_list(Module* m) {
    for (Module::global_iterator GI = m->global_begin(); GI != m->global_end(); ++GI) {
        if ((GI)->getName().str().find("_global_") != std::string::npos) {
            gv_list.push_back(&*GI);  // LLVM >= 4.0 requires explicit dereference
        }
    }
}

/*
 * Declare private backup buffers (_bak and _isDirty) for global variables
 * that appear in write-after-read (WAR) dependencies inside each function.
 */
void AlpacaModulePass::declare_priv_buffers(func_vals_map WARinFunc) {
    for (func_vals_map::iterator FIT = WARinFunc.begin(); FIT != WARinFunc.end(); ++FIT) {
        val_vec WARs = (FIT)->second;
        for (val_vec::iterator VIT = WARs.begin(); VIT != WARs.end(); ++VIT) {
            GlobalVariable* gv = cast<GlobalVariable>(*VIT);
            Value* bak = m->getNamedValue(gv->getName().str() + "_bak");

            // Declare a new backup global variable if not already present
            if (bak == NULL) {
                GlobalVariable* priv_buffer = new GlobalVariable(*m,
                    gv->getType()->getContainedType(0), false,
                    gv->getLinkage(), 0, gv->getName() + "_bak", gv);
                priv_buffer->copyAttributesFrom(gv);
                priv_buffer->setSection(".nv_vars");
                priv_buffer->setInitializer(gv->getInitializer());

                // If this is an array, also declare its corresponding isDirty array
                if (isArray(gv)) {
                    GlobalVariable* global_isDirty = new GlobalVariable(*m,
                        ArrayType::get(Type::getInt16Ty(m->getContext()),
                            gv->getType()->getContainedType(0)->getArrayNumElements()),
                        false, gv->getLinkage(), 0, gv->getName() + "_isDirty", gv);
                    global_isDirty->copyAttributesFrom(gv);
                    global_isDirty->setSection(".nv_vars");
                    ConstantAggregateZero* zeroInit =
                        ConstantAggregateZero::get(global_isDirty->getType()->getContainedType(0));
                    global_isDirty->setInitializer(zeroInit);
                }
            }
        }
    }
}

/*
 * Main module-level entry point for the Alpaca pass.
 */
bool AlpacaModulePass::runOnModule(Module &M) {
    m = &M;
    set_write_to_gbuf();
    set_my_memset();
    declare_globals();

    get_gv_list(m);

    AnalyzeTasks* AT = new AnalyzeTasks(this);
    AT->runTaskAnalysis(M);
    func_vals_map WARinFunc = AT->getWARinFunc();

    declare_priv_buffers(WARinFunc);

    TransformTasks* TT = new TransformTasks(this, m, write_to_gbuf);
    TT->runTransformation(WARinFunc);

    set_commit_buffer(AT->getMaxCommitSize());
    set_clear_isDirty_function();
    return true;
}

/*
 * Allocate and initialize the commit buffer used at runtime.
 * It contains three arrays: source addresses, destination addresses, and sizes.
 */
void AlpacaModulePass::set_commit_buffer(uint64_t commitSize) {
    errs() << "max commit size: " << commitSize << "\n";

    // (1) Source address list
    GlobalVariable* data_src = new GlobalVariable(*m,
        ArrayType::get(Type::getInt8PtrTy(m->getContext()), commitSize),
        false, GlobalValue::ExternalLinkage, 0, "data_src",
        m->getNamedGlobal("data_base"));
    data_src->setAlignment(2);
    data_src->setSection(".nv_vars");
    std::vector<Constant*> arr;
    Constant* initData = ConstantArray::get(
        ArrayType::get(Type::getInt8PtrTy(m->getContext()), commitSize),
        ArrayRef<Constant*>(arr));
    data_src->setInitializer(initData);

    // (2) Destination address list
    GlobalVariable* data_dest = new GlobalVariable(*m,
        ArrayType::get(Type::getInt8PtrTy(m->getContext()), commitSize),
        false, GlobalValue::ExternalLinkage, 0, "data_dest",
        m->getNamedGlobal("data_base"));
    data_dest->setAlignment(2);
    Constant* init_data_dest = ConstantArray::get(
        ArrayType::get(Type::getInt8PtrTy(m->getContext()), commitSize),
        ArrayRef<Constant*>(arr));
    data_dest->setInitializer(init_data_dest);
    data_dest->setSection(".nv_vars");

    // (3) Buffer size list
    GlobalVariable* data_size = new GlobalVariable(*m,
        ArrayType::get(Type::getInt16Ty(m->getContext()), commitSize),
        false, GlobalValue::ExternalLinkage, 0, "data_size",
        m->getNamedGlobal("data_base"));
    data_size->setAlignment(2);
    Constant* init_data_size = ConstantArray::get(
        ArrayType::get(Type::getInt16Ty(m->getContext()), commitSize),
        ArrayRef<Constant*>(arr));
    data_size->setInitializer(init_data_size);
    data_size->setSection(".nv_vars");
}

/*
 * Define the function clear_isDirty() that resets all _isDirty flags to zero.
 * LLVM >= 5.0 requires creating a FunctionType and using getOrInsertFunction(FunctionType).
 */
void AlpacaModulePass::set_clear_isDirty_function() {
    LLVMContext &Ctx = m->getContext();
    Type *VoidTy = Type::getVoidTy(Ctx);

    FunctionType *FT = FunctionType::get(VoidTy, false);
    FunctionCallee Callee = m->getOrInsertFunction("clear_isDirty", FT);
    Function *clear_isDirty = cast<Function>(Callee.getCallee());
    clear_isDirty->setCallingConv(CallingConv::C);

    // Create the entry basic block
    BasicBlock* block = BasicBlock::Create(Ctx, "entry", clear_isDirty);

    // For each _isDirty variable, call my_memset() to zero it out
    for (gv_vec::iterator GI = gv_list.begin(); GI != gv_list.end(); ++GI) {
        if ((*GI)->getName().str().find("_isDirty") != std::string::npos) {
            // Bitcast array pointer to i8*
            BitCastInst* arraybc = new BitCastInst(*(*GI),
                Type::getInt8PtrTy(Ctx), "", block);

            // Compute the size using GEP + cast
            val_vec arrayRef;
            arrayRef.push_back(ConstantInt::get(Type::getInt16Ty(Ctx), 1, false));
            Value* size = GetElementPtrInst::CreateInBounds(
                Constant::getNullValue((*GI)->getType()),
                ArrayRef<Value*>(arrayRef), "", block);
            Value* sizei = CastInst::Create(CastInst::getCastOpcode(size, false,
                Type::getInt16Ty(Ctx), false), size,
                Type::getInt16Ty(Ctx), "", block);

            // Arguments for my_memset(ptr, 0, size)
            Value* zero = ConstantInt::get(Type::getInt16Ty(Ctx), 0);
            val_vec args;
            args.push_back(arraybc);
            args.push_back(zero);
            args.push_back(sizei);
            CallInst::Create(array_memset, ArrayRef<Value*>(args), "", block);
        }
    }
    ReturnInst::Create(Ctx, block);
}

/*
 * Declare the runtime pre-commit function: write_to_gbuf().
 * NOTE: LLVM >= 5.0 requires constructing a FunctionType explicitly.
 */
void AlpacaModulePass::set_write_to_gbuf() {
    LLVMContext &Ctx = m->getContext();
    Type *VoidTy = Type::getVoidTy(Ctx);
    std::vector<Type*> argTypes = {
        Type::getInt8PtrTy(Ctx),
        Type::getInt8PtrTy(Ctx),
        Type::getInt16Ty(Ctx)
    };
    FunctionType *FT = FunctionType::get(VoidTy, argTypes, false);
    FunctionCallee Callee = m->getOrInsertFunction("write_to_gbuf", FT);
    write_to_gbuf = cast<Function>(Callee.getCallee());
}

/*
 * Declare the customized memset function (my_memset).
 */
void AlpacaModulePass::set_my_memset() {
    LLVMContext &Ctx = m->getContext();
    Type *VoidTy = Type::getVoidTy(Ctx);
    std::vector<Type*> argTypes = {
        Type::getInt8PtrTy(Ctx),
        Type::getInt16Ty(Ctx),
        Type::getInt16Ty(Ctx)
    };
    FunctionType *FT = FunctionType::get(VoidTy, argTypes, false);
    FunctionCallee Callee = m->getOrInsertFunction("my_memset", FT);
    array_memset = cast<Function>(Callee.getCallee());
}

/*
 * Declare library-level global variables used by Alpaca.
 */
void AlpacaModulePass::declare_globals() {
    Constant* zeroVal = ConstantInt::get(Type::getInt16Ty(m->getContext()), 0);

    GlobalVariable* num_dirty_gv = new GlobalVariable(*m,
        Type::getInt16Ty(m->getContext()), false, GlobalValue::CommonLinkage,
        0, Twine("num_dirty_gv"),
        m->getNamedGlobal("data_src_base"),
        GlobalValue::ThreadLocalMode::NotThreadLocal, 0, true);
    num_dirty_gv->setInitializer(zeroVal);

    GlobalVariable* numBoots = new GlobalVariable(*m,
        Type::getInt16Ty(m->getContext()), false, GlobalValue::CommonLinkage,
        0, Twine("_numBoots"),
        m->getNamedGlobal("data_src_base"),
        GlobalValue::ThreadLocalMode::NotThreadLocal, 0, true);
    numBoots->setInitializer(zeroVal);
}

/*
 * Standard LLVM pass registration.
 */
char AlpacaModulePass::ID = 0;
RegisterPass<AlpacaModulePass> X("alpaca", "Alpaca Pass");
