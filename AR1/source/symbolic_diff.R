## Test symbolic differentiation in R

trig.exp <- expression(sin(cos(x + y^2)))
D.sc <- D(trig.exp, "x")
all.equal(D(trig.exp[[1]], "x"), D.sc)

Dx <- deriv(trig.exp, "x")
Dx2 <- D(trig.exp, "x")
x <- 1
eval(Dx)
eval(Dx2)

dxy <- deriv(trig.exp, c("x", "y")) 
x <- 1
y <- 1
eval(dxy)
eval(D.sc)
