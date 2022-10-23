use std::f64::consts::PI;
use crate::gcmrust::{channels::base_channel, data_models::base_partition::{Partition, self}};

pub struct RidgeChannel {
    pub rho : f64,
    pub alpha : f64,
    pub gamma : f64,
}



// 

impl base_channel::ChannelWithExplicitHatOverlapUpdate for RidgeChannel {
    fn update_hatoverlaps(&self, m : f64, q : f64, v : f64) -> (f64, f64, f64) {
        
        let mhat = (self.alpha * self.gamma.sqrt()) / (1.0 + v);
        let vhat =  self.alpha / (1.0 + v);
        let qhat = self.alpha * (self.rho + q - 2.0 * m ) / (1.0 + v).powi(2);

        return (mhat, qhat, vhat);
    }
}

