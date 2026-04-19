def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    x = x0
    for i in range(steps):
        gradient = 2 * a * x + b  
        x = x - (lr * gradient)
        
    return x