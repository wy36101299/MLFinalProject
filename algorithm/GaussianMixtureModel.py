# -*- coding: UTF-8 -*-

import math
import numpy as np

class GaussianMixtureModel(object):
    '''
    GaussianMixtureModel
    '''

    def initialize(self,K, data):
        #初始化 mu,sigma,pi

        # d -> dimensions
        d = data.shape[1]

        # initialize the mu randomly 0-10
        mu = np.random.rand(K,d)*10

        # initialize the sigma
        sigma = []
        for k in range(K):
            sigma.append(np.cov(data.T))

        # initialize the pi
        sum_pi = 1.0
        pi = np.zeros(K)
        pi += sum_pi/K
        return mu, sigma, pi

    def e_step(self,K,mu,sigma,pi,data):
        #用現有的 mu,sigma,pi 估計該點屬於K群的機率

        N = len(data)
        r = np.zeros((N,K))

        for i in range(N):
            for k in range(K):
                r[i][k] = (pi[k]*self.prior_prob(data[i],mu[k],sigma[k]))/self.prior_prob_sigmak(K,mu,sigma,pi,data[i])

        return r

    def m_step(self,r, K, data):
        # 用該點屬於K群的機率 估計mu,sigma,pi

        # update mu
        N = len(data)
        N_k = np.zeros(K)
        d = data.shape[1]
        new_mu = np.zeros((K,d))

        for k in range(K):
            for n in range(N):
                N_k[k] += r[n][k]
                new_mu[k] += (r[n][k]*data[n])

            new_mu[k] /= N_k[k]

        # update sigma
        new_sigma = np.zeros((K,d,d))
        for k in range(K):
            for n in range(N):
                xn = np.zeros((1,2))
                mun = np.zeros((1,2))
                xn += data[n]
                mun += new_mu[k]
                x_mu = xn - mun
                new_sigma[k] += (r[n][k]*x_mu*x_mu.T)
            new_sigma[k] /= N_k[k]

        # update pi
        new_pi = np.zeros(3)
        for k in range(K):
            new_pi[k] += (N_k[k]/N)

        return new_mu, new_sigma, new_pi

    def likelihood(self,K,mu,sigma,pi,data):
        # 計算log likelihood 每個點對應每個K群的機率(取log)
        N = len(data)
        log_score = 0.0
        for n in range(N):
            log_score += np.log(self.prior_prob_sigmak(K,mu,sigma,pi,data[n]))
        return log_score

    def prior_prob_sigmak(self,K,mu,sigma,pi,data):
        # 計算後驗機率*k群機率

        pb = 0.0
        for k in range(K):
            pb += pi[k]*self.prior_prob(data,mu[k],sigma[k])

        return pb

    def prior_prob(self,x,mu,sigma):
        # 計算後驗機率

        score = 0.0

        x_mu = np.matrix(x - mu)
        inv_sigma = np.linalg.inv(sigma)
        det_sqrt = np.linalg.det(sigma)**0.5
        d = len(x)
        norm_const = 1.0/((2*np.pi)**(d/2)*det_sqrt)
        exp_value = math.pow(math.e,-0.5 * (x_mu * inv_sigma * x_mu.T))
        score = norm_const * exp_value

        return score

    def predict(self,data,K,mu_k,sigma_k):
        # 預測屬於哪一群

        label = []
        for i in range(len(data)):
            prior_prob_k = [self.prior_prob(data[i],mu_k[k],sigma_k[k]) for k in range(K)]
            label.append(prior_prob_k.index(max(prior_prob_k)))
        return label

    def gmm(self,K,data):
        # gmm主函數

        mu, sigma, pi = self.initialize(K,data)
        log_score = self.likelihood(K, mu, sigma, pi, data)
        threshold = 0.001
        i = 0
        max_iter = 500
        while i < max_iter:
            # expectation step
            r = self.e_step(K,mu,sigma,pi,data)

            # maximization step
            mu, sigma, pi = self.m_step(r, K, data)

            new_log_score = self.likelihood(K, mu, sigma, pi, data)
            if abs(new_log_score - log_score) < threshold:
                break
            log_score = new_log_score

            i += 1
        return mu,sigma
