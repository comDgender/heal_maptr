from common import ckpt_dir, task_name, training_step
from data import val_batch_processor, val_data_loader
from model import test_model


def update_loss(metrics, batch, model_outs):
    mean_losses = {}
    for loss_name, loss in model_outs.items():
        mean_loss = sum(loss) / len(loss)
        mean_losses[loss_name] = mean_loss
    for metric in metrics:
        metric.update(mean_losses)


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=20,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)

val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=5000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=20,
)

bn_callback = dict(
    type="FreezeBNStatistics",
)

grad_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=0.1,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=False,
    mode="max",
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    # init_with_train_model=True + val_model=test_model: Validation
    # deepcopies test_model (a clean float Motr without criterion) and
    # loads train_model's state_dict, so post_process outputs tracks
    # correctly without the training-side stateful pollution.
    init_with_train_model=True,
    val_interval=1,
    val_on_train_end=True,
    log_interval=200,
    val_model=test_model,
)
