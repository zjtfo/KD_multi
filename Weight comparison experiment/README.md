
In the weight comparison experiment, only modifications need to be made to the third ablation experiment(消融3 -> without confidence_aware) in the ablation experiments. Change transfer_multihop_loss_ce and transfer_logical_loss_ce to 0.1, 0.2, ..., up to 1.0 sequentially.

