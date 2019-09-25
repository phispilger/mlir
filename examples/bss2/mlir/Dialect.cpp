#include "bss2/Dialect.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace bss2 {

Dialect::Dialect(mlir::MLIRContext* context) : mlir::Dialect("bss2", context)
{
	addOperations<InvertOp>();
}


void InvertOp::build(mlir::Builder* builder, mlir::OperationState* state, mlir::Value* value)
{
	state->operands.push_back(value);
}

mlir::LogicalResult InvertOp::verify()
{
	return mlir::success();
}


namespace detail {

/**
 * Fold invert(invert(x)) -> x.
 */
struct SimplifyRedundantInvert : public mlir::OpRewritePattern<InvertOp>
{
	/// We register this pattern to match every bss2.invert in the IR.
	/// The "benefit" is used by the framework to order the patterns and process
	/// them in order of profitability.
	SimplifyRedundantInvert(mlir::MLIRContext* context) :
	    OpRewritePattern<InvertOp>(context, /* benefit */ 1)
	{}

	/// This method is attempting to match a pattern and rewrite it. The rewriter
	/// argument is the orchestrator of the sequence of rewrites. It is expected
	/// to interact with it to perform any changes to the IR from here.
	mlir::PatternMatchResult matchAndRewrite(
	    InvertOp op, mlir::PatternRewriter& rewriter) const override
	{
		// Look through the input of the current invert.
		mlir::Value* invertInput = op.getOperand();
		InvertOp invertInputOp = llvm::dyn_cast_or_null<InvertOp>(invertInput->getDefiningOp());

		// If the input is defined by another Invert, bingo!
		if (!invertInputOp) {
			return matchFailure();
		}

		// Use the rewriter to perform the replacement
		rewriter.replaceOp(op, {invertInputOp.getOperand()}, {invertInput});
		return matchSuccess();
	}
};

} // namespace detail

void InvertOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList& results, mlir::MLIRContext* context)
{
	results.insert<detail::SimplifyRedundantInvert>(context);
}

} // namespace bss2
