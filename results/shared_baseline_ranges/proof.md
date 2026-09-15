# Sharp support bounds for the shared baseline

Fix three scalar maps $h,Y,b$. Both contrasts use exactly the same baseline
map $b$, with $C=D=((1,0),(-1,0))$; deleting the common zero column gives
$C=D=(1,-1)^T$. Define the centered comparison

\[
p_x(i,j)=\tfrac12\operatorname{sgn}(x(W_i)-x(W_j)),
\]

where the sign is zero at ties. Let

\[
\begin{aligned}
S&=[p_h(0,1)-p_b(0,1)][p_Y(0,1)-p_b(0,1)],\\
Q&=\tfrac12[\Delta p_h-\Delta p_b][\Delta p_Y-\Delta p_b],
\qquad\Delta p_x=p_x(0,1)-p_x(0,2),\\
U&=\frac16\sum_{i,j,k\text{ distinct}}
 [p_h(i,j)-p_b(i,j)][p_Y(i,k)-p_b(i,k)].
\end{aligned}
\]

**Theorem.** For every assignment of real map values, including every tie
pattern, the sharp containing intervals are

\[
S\in[-1/4,1],\qquad Q\in[-1/2,2],\qquad U\in[-1/6,1/3].
\]

Their widths are $5/4,5/2,1/2$, respectively. Every endpoint is attainable
under a common IID trajectory law. If the map identity $h=b$ is known from
the map definitions, all three kernels are identically zero and have width zero.

**Proof for S.** Put $x=p_h(0,1)$, $y=p_Y(0,1)$, $z=p_b(0,1)$.
All three lie in $\{-1/2,0,1/2\}$. When $z=1/2$, both factors in
$(x-z)(y-z)$ lie in $[-1,0]$; when $z=-1/2$, both lie in $[0,1]$.
The product is in $[0,1]$ in either case. When $z=0$, it lies in
$[-1/4,1/4]$. This proves the asserted interval, including ties. Dropping ties
would incorrectly discard the negative endpoint for the stated application.

**Proof for Q.** Put $u=\Delta p_h,v=\Delta p_Y,w=\Delta p_b$, all in
$[-1,1]$. For fixed $w$, a bilinear function on the square
$(u,v)\in[-1,1]^2$ attains its extreme values at corners. Therefore

\[
w^2-1\le (u-w)(v-w)\le (1+|w|)^2.
\]

Dividing by two and using $|w|\le1$ proves the interval. Enlarging the set of
feasible triples to this square is safe; the endpoint witnesses below prove it
does not enlarge either final endpoint.

**Proof for U.** Write the scalar six-role kernel as

\[
k(x,y)=\frac16\sum_{i,j,k\text{ distinct}}p_x(i,j)p_y(i,k).
\]

For completeness, let $s_{ij}=\operatorname{sgn}(x_i-x_j)$ and
$t_{ij}=\operatorname{sgn}(y_i-y_j)$. Direct expansion gives

\[
24k=s_{01}t_{02}+s_{02}t_{01}-s_{01}t_{12}-s_{12}t_{01}
     +s_{02}t_{12}+s_{12}t_{02}.
\]

Reorder the three points so that $x_0\le x_1\le x_2$, permissible because
$k$ is symmetric in the observations. With strict inequalities this is
$-2t_{02}$. With $x_0=x_1<x_2$ it is $-(t_{02}+t_{12})$; with
$x_0<x_1=x_2$ it is $-(t_{01}+t_{02})$. With three equal $x$'s it is
zero. Thus $|k(x,y)|\le1/12$. Setting $y=x$ gives $k(x,x)=1/12$ for
every nonconstant triple, and zero for the constant triple.

Bilinearity gives

\[
U=k(h,Y)-k(h,b)-k(b,Y)+k(b,b).
\]

If the baseline triple is nonconstant, its last term is exactly $1/12$.
Bounding each of the other three terms by $1/12$ yields
$-1/6\le U\le1/3$. If the baseline triple is constant, every kernel term
containing $b$ vanishes, leaving $U=k(h,Y)\in[-1/12,1/12]$, which is
contained in the asserted interval.

**Sharpness.** The rows below give actual values of each map on trajectories
$(W_0,W_1,W_2)$. The third trajectory is unused by S.

| Endpoint | h values | Y values | b values |
|---|---|---|---|
| S = -1/4 | (0,1,0) | (1,0,0) | (0,0,0) |
| S = 1 | (2,1,0) | (2,1,0) | (1,2,0) |
| Q = -1/2 | (1,0,2) | (1,2,0) | (0,0,0) |
| Q = 2 | (1,2,0) | (1,2,0) | (1,0,2) |
| U = -1/6 | (0,1,2) | (1,2,0) | (0,2,1) |
| U = 1/3 | (2,1,0) | (2,1,0) | (1,0,2) |

For each row, use a trajectory space with three atoms carrying those three
coordinate triples and put positive probability on every atom. Independent
draws from this law realize the displayed ordered triple with positive
probability. Consequently these are sharp support intervals within the IID
model itself. If $h=b$ as maps, every left contrast vanishes pointwise,
establishing the zero-width claim.

## Finite verification and concentration

Every ordered triple of real values induces exactly one of 13 weak orders.
The integer encodings use consecutive labels starting at zero. The executable
`exact_checks.py` enumerates all $13^3=2,197$ independent map-order triples,
all 169 scalar map pairs, and the 169 declared-identity cases. Arithmetic uses
`fractions.Fraction`; the support extrema, bilinear expansion and endpoint
witnesses are checked exactly. `test_study.py` also compares all 2,197 cases
against the released generic matrix kernel, independently of this proof.

The intervals are deterministic consequences of the map relations. They may
replace the generic widths in the existing Hoeffding, hybrid mean, and pooled-U
variance formulas under the same fixed-map, conditional IID assumptions and
error allocations. For the asymmetric intervals, subtracting the known lower
endpoint moves the data to $[0,W]$ without changing the variance or the
estimation error. No formula requires an interval centered at zero. The widths
must be justified structurally, not estimated from the observed sample range.

The old universal comparison using generic widths
$W_U/W_S=1/3$ remains correct for those generic widths. It must not silently
be interpreted as using the sharp widths, whose ratio is $2/5$.
Using the same argument $q>2M/3$, the latter gives
$h(1/2,q,\alpha)/h(5/4,M,\alpha-\delta)<\sqrt6/5$.
This concern does not affect the fixed-budget numerical comparison below.

## What improves on exactly the existing data

Both shared-score widths decrease by a factor $5/8$. Thus the entire pure
range allocation objective decreases by $5/8$ at every feasible allocation;
its minimizers are unchanged. At budget 73,728, the existing integer allocation
$M=13,086,n=15,852$ remains optimal for that objective. All original draw
positions, centers and variance estimates stay fixed. Full-U width decreases
by $3/4$, so its pure range radius falls by 25%.

For boosting_31, the independent replay gives:

| Rule and fixed allocation | Generic total radius | Structural total radius | New lower bound |
|---|---:|---:|---:|
| Original shared, hybrid | 0.0157480213475351 | 0.0119266246846745 | -0.0115451549581120 |
| Allocated shared, hybrid | 0.0121261237918128 | 0.00958947507154764 | -0.00771174600778578 |
| Allocated shared, pure range | 0.0836462810120319 | 0.0522789256325199 | -0.0504011965687581 |
| Direct triple U, hybrid | 0.000755407868251283 | 0.000653157891458074 | 0.000216593085104426 |
| Complete U, variance | 0.000692417110582742 | 0.000601138014597516 | 0.000246219565642549 |
| Complete U, pure range | 0.00677428719014945 | 0.00508071539261209 | -0.00423335781237203 |

The allocated shared hybrid radius falls 20.9189% solely from widths; the pure
range radius falls exactly 37.5%. This is distinct from the previously reported
23.00% hybrid improvement from the old allocation to the new allocation using
the old generic widths. The allocated sharp hybrid radius is 39.107% below the
original generic hybrid radius; that combined number includes two operations
and should not be attributed to the width theorem alone.

All shared rules still have no positive candidate lower bound. The complete-U
variance rule newly certifies boosting_7: its lower bound increases from
$-0.0000799943199261226$ to $0.0000112847760591031$, giving four certified
candidates instead of three. Its selected predictor remains boosting_31, the
minimum selection-period MSE among certified candidates. The direct triple rule
also continues to select boosting_31. All chosen predictors are unchanged, so
this replay adds no observed or independently confirmed predictive utility.

The exact population quantities for boosting_31 are

\[
E[S]=0.0099090318388564,\quad
\gamma=0.009056899048860137,\quad
\theta=0.0008521327899962623.
\]

Thus the exact interaction share is
$\gamma/E[S]=0.9140044351603771$. Dividing an estimated correction by a
sample raw score answers a different question and can be distorted by Monte
Carlo error. The 91.4004% statement uses the population denominator explicitly.

This is a post-exposure diagnostic on already inspected cohorts. It establishes
sharper support and smaller valid radii under the stated sampling model. It does
not supply new independent calibration, new forecasting data, or new loss gains.
