#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
class Builder;
} // namespace mlir

namespace bss2 {

/**
 * Declaration of the BSS2 dialect.
 * It inherits from mlir::Dialect and registers custom operations and types (in its constructor).
 * It can also override general behavior of dialects exposed as virtual method, for example
 * regarding verification and parsing/printing.
 */
class Dialect : public mlir::Dialect
{
public:
	explicit Dialect(mlir::MLIRContext* context);
};


class InvertOp
    : public mlir::Op<
          InvertOp,
          mlir::OpTrait::OneOperand,
          mlir::OpTrait::OneResult,
          mlir::OpTrait::HasNoSideEffect>
{
public:
	static llvm::StringRef getOperationName() { return "bss2.invert"; }

	/**
	 * Operation can add custom verification beyond the traits they define.
	 */
	mlir::LogicalResult verify();

	/**
	 * Interface to mlir::Builder::create<InvertOp>(...).
	 * The `bss2.invert` operation accepts a single argument and returns the inverted value as only
	 * result.
	 */
	static void build(mlir::Builder* builder, mlir::OperationState* state, mlir::Value* value);

	/**
	 * Register rewrite pattern by the Canonicalization framework.
	 */
	static void getCanonicalizationPatterns(
	    mlir::OwningRewritePatternList& results, mlir::MLIRContext* context);

	using Op::Op;
};

} // namespace bss2
