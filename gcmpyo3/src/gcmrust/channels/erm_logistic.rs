use optimization::{Minimizer, GradientDescent, NumericalDifferentiation, Func};
use roots::{find_root_newton_raphson, SimpleConvergency, find_root_brent};
use crate::gcmrust::channels::base_channel;

static PROXIMAL_TOLERANCE : f64 = 0.001;

fn logistic_loss(z : f64) -> f64 {
    return (1.0 + (-z).exp()).ln();
}


fn logistic_loss_derivative(y : f64, z : f64) -> f64 {
    if y * z > 0.0 {
        let x = (- y * z).exp();
        return - y * x / (1.0 + x);
    }
        
    else {
        return - y / ((y * z).exp() + 1.0);
    }
        
}

fn logistic_loss_second_derivative(y : f64, x : f64) -> f64 {
    if (y * x).abs() > 500.0 {
        if y * x > 0.0 { return 1.0 / 4.0 * (-y * x).exp(); }
        else { return 1.0 / 4.0 * (y * x).exp(); }       
    }
    else {
        return 1.0 / (4.0 * (y * x / 2.0).cosh().powi(2));
    }
    /*    
    let expo = (x * y).exp();
    return 1.0 * expo / (1.0 + expo).powi(2) ;
    */
}

//

fn moreau_logistic_loss(x : f64, y : f64, omega : f64, v : f64) -> f64 {
    return (x - omega).powi(2) / (2.0 * v) + logistic_loss(y * x);
}

fn moreau_logistic_loss_derivative(x : f64, y : f64, omega : f64, v : f64) -> f64 {
    // derivative with respect to x
    return (x - omega) / v + logistic_loss_derivative(y, x);
}

fn moreau_logistic_loss_second_derivative(x : f64, y : f64, omega : f64, v : f64) -> f64 {
    return (1.0/ v) + logistic_loss_second_derivative(y, x);
}


fn iterative_proximal_logistic_loss(omega : f64, v : f64, y : f64) -> f64 {
    let mut x = omega;
    for i in 0..100 {
        x = omega - v * logistic_loss_derivative(y, x);
    }
    return x;
}

fn proximal_logistic_loss(omega : f64, v : f64, y : f64) -> f64 {
    
    // USES BRENT

    let mut convergency = SimpleConvergency { eps:1e-15f64, max_iter:30 };
    let root = find_root_brent(omega - 50.0 * v, omega + 50.0 * v, |x : f64| -> f64 {moreau_logistic_loss_derivative(x, y, omega, v)}, &mut convergency);
    return match root {
        Err(e) => iterative_proximal_logistic_loss(omega, v, y),
        Ok(v )         => v,
    };
    
    /*
    // USES NEWTON RAPHSON

    let mut convergency = SimpleConvergency { eps: PROXIMAL_TOLERANCE, max_iter : 1000 };
    let root = find_root_newton_raphson(omega as f64, |x : f64| -> f64 {moreau_logistic_loss_derivative(x, y, omega, v)}, |x : f64| -> f64 {moreau_logistic_loss_second_derivative(x, y, omega, v)}, &mut convergency);
    return match root {
        Err(e) => iterative_proximal_logistic_loss(omega, v, y),
        Ok(v )         => v,
    };
     */
    
    /* 
    // USES GRADIENT DESCENT 

    let minimizer = GradientDescent::new();
    let to_minimize = NumericalDifferentiation::new(Func(|x: &[f64]| -> f64 {
        moreau_logistic_loss(x[0], y, omega, v)
    }));

    let x_sol = minimizer.minimize(&to_minimize, vec![omega - 100.0 * v, omega + 100.0 * v]);
    return x_sol.position[0];    
    */
}

pub struct ERMLogistic {

}

impl base_channel::Channel for ERMLogistic {
    fn f0(&self, y : f64, omega : f64, v : f64) -> f64 {
        let lambda_star  = proximal_logistic_loss(omega, v, y);
        return (lambda_star - omega) / v;
    }

    fn df0(&self, y : f64, omega : f64, v : f64) -> f64 {
        let lambda_star  = proximal_logistic_loss(omega, v, y);
        let dlambda_star = 1.0 / (1.0 + v * logistic_loss_second_derivative(y, lambda_star));
        return (dlambda_star - 1.0) / v;
    }
}