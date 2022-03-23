#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Dialect/SCF/SCF.h>
class Mem2RegRewrite{
protected:
    virtual bool isLoad(mlir::Operation* op) const = 0;
    virtual int loadId(mlir::Operation* op) const = 0;
    virtual bool isStore(mlir::Operation* op) const = 0;
    virtual int storeId(mlir::Operation* op) const = 0;
    virtual bool storeValue(mlir::Operation* op) const = 0;
public:
    void mem2regOp(mlir::SmallVector<mlir::Value>& values, mlir::PatternRewriter& rewriter, mlir::scf::IfOp op) const{
        mlir::SmallVector<mlir::Type> ifReturn;
        for(auto ty: op->getResultTypes()){
            ifReturn.push_back(ty);
        }
        for(auto v: values){
            ifReturn.push_back(v.getType());
        }
        mlir::PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(op);
        auto new_if = rewriter.create<mlir::scf::IfOp>(op->getLoc(), op.getCondition(), true);
        op.getThenRegion().takeBody(new_if.getThenRegion());
        if(op.elseBlock()){
            op.getElseRegion().takeBody(new_if.getElseRegion());
        }
        mem2regBlock({}, new_if.thenBlock(), rewriter, values);
        mem2regBlock({}, new_if.elseBlock(), rewriter, values);
        for(auto i=0; i<values.size(); i++){
            values[i]=new_if->getResult(i+op->getNumResults());
        }
        rewriter.replaceOp(op, new_if->getOpResults().take_front(op.getNumResults()));
    }
// TODO: take this part out as a standalone mem2reg rewriter. 
    void mem2regBlock(mlir::TypeRange types, mlir::Block* block, mlir::PatternRewriter& rewriter, mlir::SmallVector<mlir::Value> blockArgs) const {
        if(blockArgs.size()==0){
            auto original_id = block->getNumArguments();
            for(auto ty: types){
                blockArgs.push_back(block->addArgument(ty, mlir::UnknownLoc::get(rewriter.getContext())));
            }
        }

        // go forward.
        mlir::SmallVector<mlir::Operation*> all_ops;
        for(auto& op: block->getOperations()){
            all_ops.push_back(&op);
        }
        for(auto pop: all_ops){
            auto& op = *pop;
            if(this->isLoad(pop)){
                auto load_id = this->
                rewriter.replaceOp(&op, blockArgs[load_id]);
            }else if(auto attr = op.getAttr(ISQ_PURE_STORE).dyn_cast_or_null<mlir::IntegerAttr>()){
                auto store_id = attr.getInt();
                auto operand_id = op.getAttrOfType<mlir::IntegerAttr>(ISQ_PURE_STORE_OPERAND).getInt();
                auto stored_value = op.getOperand(operand_id);
                blockArgs[store_id] = stored_value;
                rewriter.eraseOp(&op);
            }else if (op.mightHaveTrait<mlir::OpTrait::IsTerminator>()){
                do{
                    auto last = &op;
                    mlir::PatternRewriter::InsertionGuard guard(rewriter);
                    mlir::SmallVector<mlir::Value> allArgs;
                    for(auto arg: last->getOperands()){
                        allArgs.push_back(arg);
                    }
                    for(auto blockArg: blockArgs){
                        allArgs.push_back(blockArg);
                    }
                    mlir::OperationState state(last->getLoc(), last->getName(), allArgs, last->getResultTypes(), last->getAttrs(), last->getSuccessors(), {});
                    rewriter.setInsertionPointAfter(last);
                    rewriter.createOperation(state);
                    rewriter.eraseOp(last);
                }while(0);
            }else if(auto new_op = llvm::dyn_cast<mlir::scf::IfOp>(op)){
                mem2regOp(blockArgs, rewriter, new_op);
            }
        }
    }

}