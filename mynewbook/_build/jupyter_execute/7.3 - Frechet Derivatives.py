#!/usr/bin/env python
# coding: utf-8

# # Frechet Derivatives - Overview Jan 23, 2023

# In[1]:


import numpy as np

np.mean([1, 2, 3])


# \begin{gather*}
# a_1=b_1+c_1\\
# a_2=b_2+c_2-d_2+e_2
# \end{gather*}
# 
# \begin{align}
# a_{11}& =b_{11}&
#   a_{12}& =b_{12}\\
# a_{21}& =b_{21}&
#   a_{22}& =b_{22}+c_{22}
# \end{align}
# 
# \begin{gather*}
# \mathbb{R} + \mathcal{N} + \text{Var}(X^2) +\mathbf{Q}
# \end{gather*}

# If $T$ is Frechet differentiable, then $\delta T(x; h) = A_x h$, where $A_x$ is a bounded linear operator from $X$ to $Y$. The Frechet derivative $T': D \subseteq X \to B(X, Y)$ of $T$ is defined as the mapping that maps the point $x$ to the bounded operator $A_x$. 
# 
# We say that the Frechet derivative of $T$ is continuous at $x_0$ if the mapping $T'$ is continuous at $x_0$; this is different from say $T'(x_0)$ is continuous - which is always true. If the derivative of $T$ is continuous on some open sphere $S$, we say that $T$ is continuously Frechet differentiable on $S$.
# 
# > Note: We want $S$ to be open because the definition of continuous function requires an open ball of radius $\delta$ around the querry point $x_0$.
# 
# If $f$ is a functional, then $f'(x) \in X^*$ is called the gradient (which is denoted as $\nabla f(x)$).
# 
# Much of the theory of ordinary derivatives can be generalized to Frechet
# derivatives. For instance, the implicit function theorem and Taylor series
# have very satisfactory extensions. The interested reader should consult the references cited at the end of the chapter.

# **Claim.** If $T_1$ and $T_2$ are Frechet differentiable at $x\in D$, then $\alpha_1 T_1 + \alpha_2 T_2$ is Frechet differentiable at $x$ and $(\alpha_1 T_1 + \alpha_2 T_2)'(x) = \alpha_1 T_1'(x) + \alpha_2 T_2'(x).$ 

# **Proof.** To show that that $\alpha_1 T_1 + \alpha_2 T_2$ is Frechet differentiable, let $x \in D, h \in X$ be given. It is true that $\delta (\alpha_1 T_1 + \alpha_2 T_2)(x; h)$ exists, since
# \begin{align}
# \delta (\alpha_1 T_1 + \alpha_2 T_2)(x; h) & = \lim_{\alpha \to 0} \frac{1}{\alpha}[(\alpha_1 T_1 + \alpha_2 T_2)(x + h) - (\alpha_1 T_1 + \alpha_2 T_2)(x)]\\
# & = \lim_{\alpha \to 0} \frac{1}{\alpha}[\alpha_1 T_1(x + h) - \alpha_1 T_1(x)] + \lim_{\alpha \to 0} \frac{1}{\alpha}[\alpha_2 T_2(x + h) - \alpha_2 T_2(x)]\\
# & = \alpha_1 \delta T_1(x; h) + \alpha_2 \delta T_2(x; h).
# \end{align}
# Hence, $\alpha_1 T_1 + \alpha_2 T_2$ is Gateux differentiable. Moreover, $\delta T_1(x; h)$ and $\delta T_2(x; h)$ are linear and bounded in $h$, meaning that $\delta (\alpha_1 T_1 + \alpha_2 T_2)(x; h)$ is also linear and bounded in $h$. Finally, if we define $T: = \alpha_1 T_1 + \alpha_2 T_2$, then 
# \begin{align}
# \lim_{\lVert h \rVert \to 0} & \frac{\lVert T(x + h) - T(x) - \delta T(x; h) \rVert_Y}{\lVert h \rVert_X} \\ &\leq \lim_{\lVert h \rVert \to 0} \lvert \alpha_1 \rvert \frac{\lVert T_1(x + h) - T_1(x) - \delta T_1(x; h) \rVert_Y}{\lVert h \rVert_X} + \lim_{\lVert h \rVert \to 0} \lvert \alpha_2 \rvert \frac{\lVert T_2(x + h) - T_2(x) - \delta T_2(x; h) \rVert_Y}{\lVert h \rVert_X},
# \end{align}
# by the triangle inequality. The RHS converges to 0, since $T_1$ and $T_2$ are Frechet differentiable. We also have
# $
# \delta (\alpha_1 T_1 + \alpha_2 T_2)(x; h) = \alpha_1 \delta T_1(x; h) + \alpha_2 \delta T_2(x; h),$ meaning that
# $$
# (\alpha_1 T_1 + \alpha_2 T_2)'(x) h = \alpha_1 T_1'(x)h + \alpha_2 T_2'(x)h.
# $$
# 

# **Proposition 1.** Let $S$ be a transformation mapping an open set $D \subseteq X$ into an open set $E \subseteq Y$ and let $P$ be a transformation mapping $E$ into a normed space $Z$. Put $T = PS$ and suppose $S$ is Frechet differentiable at $x \in D$ and $P$ is Frechet differentiable at $y = S(x) \in E$. Then $T$ is Frechet differentiable at $x$ and $T'(x) = P'(y) S'(x)$. 

# **Proof.**
# > **Definition**: 
# <br>Little $o$ notation: if $f(x) \in o(g(x))$ as $x \to a$, then
# $$
# \lim_{x \to a} \frac {f(x)} {g(x)} = 0.
# $$
# <br>Big $O$ notation: if $f(x) \in O(g(x))$  as $x \to a$, then
# $$
# \lim_{x \to a}  \frac {f(x)} {g(x)} < \infty.
# $$
# For both definition, $a$ is not specified, then it is equal to 0.
# 
# For $h \in X, x+h \in D$, we have
# $$
# T(x+h)-T(x) = P[S(x+h)]-P[S(x)] = P(y+g)-P(y)
# $$
# where $g=S(x+h)-S(x)$. Thus $\lVert T(x+h)-T(x)-P'(y)g \rVert = o(\lVert g \rVert)$ and $\lVert g - S'(x)h \rVert = o(\lVert h \rVert)$. We therefore have
# $$
# \lim_{\lVert h \rVert \to 0}\frac{\lVert g - S'(x)h \rVert}{\lVert h \rVert} = 0
# \Leftrightarrow
# \lim_{\lVert h \rVert \to 0}\frac{g - S'(x)h }{\lVert h \rVert} = 0.
# $$
# Thus, by continuity and linearity of $P'(y)$, we have
# $$
# P'(y) \lim_{\lVert h \rVert \to 0}\frac{ g - S'(x)h}{\lVert h \rVert} = 0 \Leftrightarrow
# \lim_{\lVert h \rVert \to 0}\frac{ P'(y) g - P'(y) S'(x)h }{\lVert h \rVert} = 0,
# $$
# One has $\lVert g \rVert = O(\lVert h \rVert)$ as $\lVert h \rVert \to 0$ because, by the Frechet diferentiability of $S$ and the fact that since $S'(x)$ is bounded, we have
# $$
# \lim_{\lVert h \rVert \to 0}\frac{\lVert g  \rVert}{\lVert h \rVert} = \lim_{\lVert h \rVert \to 0}\frac{\lVert S'(x)h \rVert}{\lVert h \rVert} \leq \lim_{\lVert h \rVert \to 0}\frac{\lVert S'(x) \rVert \lVert h \rVert}{\lVert h \rVert} = \lVert S'(x) \rVert < \infty.
# $$
# Then:
# \begin{align}
# &\lim_{\lVert h \rVert \to 0} \frac {\lVert T(x+h)-T(x)-P'(y)S'(x)h \rVert} {\lVert h \rVert} \\
# &\leq \lim_{\lVert h \rVert \to 0} \frac {\lVert T(x+h)-T(x)-P'(y)g \rVert} {\lVert h \rVert} + \lim_{\lVert h \rVert \to 0} \frac{\lVert P'(y)g -P'(y)S'(x)h \rVert} {\lVert h \rVert}\\
# &= \lim_{\lVert h \rVert \to 0} \frac {\lVert T(x+h)-T(x)-P'(y)g \rVert} {\lVert h \rVert} = \lim_{\lVert g \rVert \to 0} \frac {\lVert P(y+g)-P(y)-P'(y)g \rVert} {\lVert g \rVert} \lim_{\lVert h \rVert \to 0} \frac{\lVert g \rVert}{\lVert h \rVert} \leq 0 \cdot \lVert S'(x) \rVert = 0.
# \end{align}
# Note that in the last line, we can change from limit of $\lVert h \rVert \to 0$ to $\lVert g \rVert \to 0$ because according to Proposition 3 of Section 7.2, $S$ is continuous at $x$, leading to
# $$
# \lim_{\lVert h \rVert \to 0} g = \lim_{\lVert h \rVert \to 0} S(x+h)-S(x) = S(x + \lim_{\lVert h \rVert \to 0}h)-S(x) = S(x) - S(x) = 0.
# $$
# 
# 
# Since Frechet differential is unique, we have
# $$
# T'(x)h = P'(y)S'(x)h.
# $$
# 
# 
# 

# > *Note*: It is also beneficial to learn the short hand argument approach through big $o$ and little $O$ notation presented in Luenberger. We know $\lVert T(x+h)-T(x)-P'(y)g \rVert = o(\lVert g \rVert)$ and $\lVert P'(y)g - P'(y)S'(x)h \rVert = o(\lVert h \rVert)$, hence
# \begin{align}
# \lVert T(x+h)-T(x)- P'(y) S'(x)h \rVert &\leq \lVert T(x+h)-T(x)-P'(y)g \rVert + \lVert P'(y)g - P'(y)S'(x)h \rVert \\
# & = o(\lVert h \rVert) + o(\lVert g \rVert)\\
# \Leftrightarrow \frac{\lVert T(x+h)-T(x)- P'(y) S'(x)h \rVert}{\lVert h \rVert} & \leq \frac{o(\lVert h \rVert)}{\lVert h \rVert} + \frac{o(\lVert g \rVert)}{\lVert g \rVert} \frac{\lVert g \rVert}{\lVert h \rVert}\\
# \end{align}
# One has $\lVert g \rVert = O(\lVert h \rVert)$ as $\lVert h \rVert \to 0$ because of the Frechet diferentiability of $S$ and the fact that since $S'(x)$ is bounded. This also implies that $\lVert g \rVert \to 0$ as $\lVert h \rVert \to 0$. Therefore, 
# $$
# \frac{\lVert T(x+h)-T(x)- P'(y) S'(x)h \rVert}{\lVert h \rVert} \leq \frac{o(\lVert h \rVert)}{\lVert h \rVert} + \frac{o(\lVert g \rVert)}{\lVert g \rVert} \frac{O(\lVert h \rVert)}{\lVert h \rVert}.\\
# $$
# Taking the limit as $\lVert h \rVert \to 0$ on both sides yield the desired result.

# **Proposition 2.** Let $T$ be Frechet diferentiable on a open domain $D$. Let $x \in D$ and suppose that $x + \alpha h \in D$ for all $0 \leq \alpha \leq 1$. Then
# $$
# \lVert T(x+h) - T(x) \rVert \leq \lVert h \rVert \sup_{0< \alpha < 1} \lVert T'(x+\alpha h) \rVert.
# $$

# **Proof.** Let $x$ and $h$ be fixed and let $y^*$ be a nonzero element of the dual space $Y^*$ aligned with the element $T(x +h) - T(x)$. Let us remind ourselves that a vector $s^* \in S^*$ is said to be aligned with a vector $s \in S$ if $s^*(s) = \lVert s^* \rVert \lVert s \rVert$.
# 
# The function $\varphi(\alpha) = y^*[T(x+\alpha h)]$ is defined on the interval $[0, 1]$ and by the chain rule, has derivative
# $$
# \varphi'(\alpha) = y^*[T'(x + \alpha h)h].
# $$
# > *Note*: This is true since $y^*$ is linear, meaning that $\delta y^*(y, k) = y^* (k)$ - or $(y^{*})' = y^*$ (by a note from Section 7.2). We can rewrite $T(x + \alpha h)= T U(\alpha)$, where $U(\alpha) = x + \alpha h$. We can show that $\delta U(\alpha; \beta)$ is linear and continuous and Frechet differentiable. Therefore, $$\delta U(\alpha; \beta) = \lim_{\gamma \to 0} \frac{1}{\gamma} [U(\alpha + \gamma \beta) -  U(\alpha)] = \beta h = U'(\alpha) \beta,$$ meaning that $U'(\alpha) = h$. 
# 
# 
# By the mean value theorem for functions of a real variable, we have 
# $$\varphi(1) - \varphi(0) = \varphi'(\alpha_0), \text{ for some } 0<\alpha_0 < 1,$$
# and hence
# $$
# \lvert y^*[T(x + h) - T(x)]\rvert \leq \lVert y^* \rVert \sup_{0<\alpha<1} \lVert T'(x+\alpha h) \rVert \lVert h \rVert,
# $$
# and since $y^*$ is aligned with $T(x + h) - T(x)$, we have 
# $
# \lVert y^* \rVert  \lVert T(x + h) - T(x)\rVert \leq \lVert y^* \rVert \lVert h \rVert \sup_{0<\alpha<1} \lVert T'(x+\alpha h) \rVert,
# $
# which means
# $$
# \lVert T(x + h) - T(x)\rVert \leq  \lVert h \rVert \sup_{0<\alpha<1} \lVert T'(x+\alpha h) \rVert.
# $$

# If $T: X \to Y$ is Frechet differentiable on an open domain $D\subseteq X$, the derivative $T'$ maps $D$ into $B(X, Y)$ an may itself be Frechet differentiable on a subset $D_t \subseteq D$. In this case, the Frechet derivative of $T'$ is called the second Frechet derivative of $T$ and is denoted by $T''.$

# > *Note*: The second Frechet derivative $T'': D_t \subset X \to B(X, B(X, Y))$, where $B(X, B(X, Y))$ is the space of bounded linear mapping between $X$ and the normed linear space $B(X, Y)$.

# **Example 1.** Let $f$ be a functional on $X = E^n$ having continuous partial derivatives up to second order. Then $f''(x_0)$ is an operator from $E^n$ to $E^n$ having matrix form
# $$
# f''(x_0) = \left[\frac{\partial^2 f(x)}{\partial x_i \partial x_j}\right]_{x = x_0},
# $$
# where $x_i$ is the $i$-th component of $x$.

# 

# **Proposition 3.** Let $T$ be twice Frechet differentiable on an open domain $D$. Let $x\in D$ and suppose that $x + \alpha h \in D$ for all $0\leq \alpha \leq 1$. Then
# $$
# \lVert T(x+h) - T(x) - T'(x)h \rVert \leq \frac{1}{2} \lVert h \rVert \sup_{0< \alpha < 1} \lVert T''(x+\alpha h) \rVert.
# $$
