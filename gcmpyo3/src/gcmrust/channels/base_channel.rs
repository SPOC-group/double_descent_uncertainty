pub trait Channel {
    /*
    Trait used for the estimators where we can compute f0, df0 (which is the case for the ERM) 
    without having to know what is the z0 of dz0.
    */
    fn f0(&self, y : f64, omega : f64, v : f64) -> f64;
    fn df0(&self, y : f64, omega : f64, v : f64) -> f64;
}

pub trait ChannelWithExplicitHatOverlapUpdate {
    fn update_hatoverlaps(&self, m : f64, q : f64, v : f64) -> (f64, f64, f64);
}