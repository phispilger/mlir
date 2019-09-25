#include "bss2/Dialect.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace cl = llvm::cl;

static cl::opt<std::string> InputFilename(
    cl::Positional, cl::desc("<input mlir file>"), cl::init("-"), cl::value_desc("filename"));

static cl::opt<bool> EnableOpt("opt", cl::desc("Enable optimizations"));

mlir::LogicalResult optimize(mlir::ModuleOp module)
{
	mlir::PassManager pm(module.getContext());
	pm.addPass(mlir::createCanonicalizerPass());
	applyPassManagerCLOptions(pm);
	auto ret = pm.run(module);
	return ret;
}

mlir::OwningModuleRef loadFileAndProcessModule(mlir::MLIRContext& context)
{
	mlir::OwningModuleRef module;
	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
	    llvm::MemoryBuffer::getFileOrSTDIN(InputFilename);

	if (std::error_code EC = fileOrErr.getError()) {
		llvm::errs() << "Could not open input file: " << EC.message() << "\n";
		return nullptr;
	}

	llvm::SourceMgr sourceMgr;
	sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

	module = mlir::parseSourceFile(sourceMgr, &context);
	if (!module) {
		llvm::errs() << "Error can't load file " << InputFilename << "\n";
		return nullptr;
	}

	if (failed(mlir::verify(*module))) {
		llvm::errs() << "Error verifying MLIR module\n";
		return nullptr;
	}

	if (EnableOpt) {
		if (failed(optimize(*module))) {
			llvm::errs() << "Module optimization failed\n";
			return nullptr;
		}
	}
	return module;
}


int main(int argc, char** argv)
{
	mlir::registerPassManagerCLOptions();
	cl::ParseCommandLineOptions(argc, argv, "bss2 compiler\n");

	mlir::registerDialect<bss2::Dialect>();

	mlir::MLIRContext context;
	auto module = loadFileAndProcessModule(context);

	module->dump();
	return 0;
}
