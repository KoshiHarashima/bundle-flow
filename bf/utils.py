# bf/utilis.pyの候補#
# make_t_grid(T=1.0, steps=50)：Eq.(12),(20)の時間積分用グリッド生成。
# lambda_schedule(step, total, start=1e-3, end=0.2)：SoftMax温度を0.001→0.2に線形/余弦で上げる（Setup記載）。
# seed_all(seed)：再現性。
# save_ckpt(path, flow, menu)/load_ckpt(path)：学習の中断・再開。
# to_device(obj, device)：Tensor/Moduleの一括デバイス移動。
# eval_revenue_argmax(V, flow, menu, t_grid)