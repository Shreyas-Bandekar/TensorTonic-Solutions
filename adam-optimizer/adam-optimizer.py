def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
   
    param, grad, m, v = np.array(param), np.array(grad), np.array(m), np.array(v)

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)

    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m, v
