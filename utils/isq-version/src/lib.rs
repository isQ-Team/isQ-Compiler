#![feature(const_option)]
#![feature(const_option_ext)]
#![feature(const_trait_impl)]

pub const ISQ_BUILD_VERSION: Option<&'static str> = option_env!("ISQ_BUILD_VERSION");
pub const ISQ_BUILD_SEMVER : Option<&'static str> = option_env!("ISQ_BUILD_SEMVER");
pub const ISQ_BUILD_REV : Option<&'static str> = option_env!("ISQ_BUILD_REV");
pub const ISQ_BUILD_FROZEN : Option<&'static str> = option_env!("ISQ_BUILD_FROZEN");


pub struct ISQVersion;

impl ISQVersion{
    pub const fn build_version()->&'static str{
        ISQ_BUILD_VERSION.unwrap_or("0.0.0")
    }
    pub const fn build_semver()->&'static str{
        ISQ_BUILD_SEMVER.unwrap_or("0.0.0+unknown.badver")
    }
    pub const fn build_rev()->&'static str{
        ISQ_BUILD_REV.unwrap_or("unknown")
    }
    pub fn build_frozen()->bool{
        ISQ_BUILD_FROZEN==Some("1")
    }
}
