df <- read.csv("wf.csv",head=T,sep=",")
plot(df$Delta_phi,df$eta,type="b")