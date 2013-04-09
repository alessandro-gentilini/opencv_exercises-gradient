old_s <- read.csv("./opencv_exercises-gradient/log.txt",sep=",",head=T)
new_s <- read.csv("./copy-opencv_exercises-gradient/log.txt",sep=",",head=T)

t.test(old_s$X.U-new_s$X.U)

library(mratios)
print(t.test.ratio(new_s$X.U,old_s$X.U))
boxplot(old_s$X.U,new_s$X.U,notch=T)

df <- read.csv("./opencv_exercises-gradient/wf.csv",head=T,sep=",")
plot(df$Delta_phi,df$eta,type="b")

rt <- read.csv("./opencv_exercises-gradient/rt_0.csv",head=T,sep=",",skip=1)
hist(rt$gradient_phase %% 360,breaks=seq(0,359))