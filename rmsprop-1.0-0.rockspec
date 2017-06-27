--[[
    Summary: Implements centered rmsprop by a slight modification to optim.rmsprop

    Author: Alireza Goudarzi
    RIKEN Brain Science Institute
    alireza.goudarzi@riken.jp
    Jun 27, 2017



]]

package = "RMSPROP"
version = "1.0-0"
source = {
   url = "./",
   tag = "v1.0",
}
description = {
   summary = "Centered rmsprop for torch optim.",
   detailed = [[
     Centered rmsprop for torch optim.
   ]],
   homepage = "https://github.com/alirezag",
   license = "MIT/X11"
}
dependencies = {
   "lua >= 5.1, < 5.4",
   "torch >= 7.0",
   "nn  >= 1.0",
   "optim >= 1.0.5",
   "dpnn = scm-1"
}


build = {
   type = "builtin",
   modules = {
      init = "init.lua",
    LogisticMap = "rmsprop.lua",
      
   },
   copy_directories = { "doc", "test" },
}