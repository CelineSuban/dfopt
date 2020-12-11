import nevergrad
import numpy as np
import random
from pdfo import newuoa
from bic import BallInACupCostFunction
import cma
import matplotlib.pyplot as plt
import optuna


class Rosenbrock:
    count = 0

    def __init__(self):
        pass

    def __call__(self, pt):
        a = [1]
        b = [100]

        diff = np.subtract(a, pt[0])
        term1 = np.square(diff)

        diffsq = pt[1] - np.square(pt[0])
        diffsq2 = np.square(diffsq)
        term2 = np.multiply(b, diffsq2)
        ans = term1 + term2

        # incrementing the global counter
        self.count += 1

        return ans

    def retcount(self):
        return self.count


# adapted from Introduction to Derivative Free Optimization by Andrew R. Conn, Katya Scheinberg, and Luis Nunes
def coordinate_search(f, oldpt, d, noiter):
    alpha = 1
    plot = []
    suc = []
    # making a maximum of 200 function evaluations
    while f.retcount() < noiter:
        successful = False

        # poll step
        # going through 4 poll vectors per point, using complete polling
        for j in range(d.shape[0]):
            if not successful:

                newpt = oldpt + alpha * d[j, :]
                # if a successful poll point is found we stop polling, and set the old point to the new point
                if f.retcount() <= noiter-2:

                    # remember you need bic.py to return the success here, so pass True to returnsuc
                    temp, suc1 = f(newpt, True)
                    temp1, suc2 = f(oldpt, True)
                    plot.append(temp1)
                    suc.append(suc2)

                    if temp < temp1:
                        oldpt = newpt
                        successful = True
                        plot.append(temp)
                        suc.append(suc1)
                        # print(suc1)
                    else:
                        plot.append(temp1)
                        suc.append(suc2)
                        # print(suc2)

        # parameter update
        # if the iteration was not successful we decrease alpha and iterate with the same point
        if not successful:
            alpha = alpha / 2

    return plot, suc


# adapted from Introduction to Derivative Free Optimization by Andrew R. Conn, Katya Scheinberg, and Luis Nunes
def directional_direct_search(f, x, d, noiter):
    alpha = 1
    oldpt = x[0]
    plot = []
    suc = []
    pts = []

    # calculating and storing the initial point's values
    oldrew, oldsuc = f(x[0], returnsuc=True)
    print("1")
    plot.append(oldrew)
    suc.append(oldsuc)
    pts.append(oldpt)

    while f.retcount() < noiter:
        successful = False

        # going through all the points in x until a successful point is found
        # search step
        for i in range(x.shape[0]):
            if f.retcount() <= noiter - 3 and successful == False:

                # remember you need bic.py to return the success here, so pass True to returnsuc
                newrew, newsuc = f(x[i], returnsuc=True)

                # plotting the better point, which is saved as oldpt
                if newrew < oldrew:
                    successful = True
                    oldpt = x[i]
                    oldrew = newrew
                    oldsuc = newsuc
                    plot.append(newrew)
                    suc.append(newsuc)
                    pts.append(x[i])
                else:
                    plot.append(oldrew)
                    suc.append(oldsuc)
                    pts.append(oldpt)

        # poll step done is search step fails
        if not successful:
            for k in range(x.shape[0]):

                # going through 4 poll vectors per point, using opportunistic polling
                for j in range(d.shape[0]):
                    if f.retcount() <= noiter - 3 and successful == False:

                        newpt = oldpt + alpha * d[j, :]

                        newrew, newsuc = f(newpt, returnsuc=True)

                        # plotting the better point, which is saved as oldpt
                        if newrew < oldrew:
                            successful = True
                            oldpt = newpt
                            oldrew = newrew
                            oldsuc = newsuc
                            plot.append(newrew)
                            suc.append(newsuc)
                            pts.append(newpt)
                        else:
                            plot.append(oldrew)
                            suc.append(oldsuc)
                            pts.append(oldpt)

        # mesh parameter update
        # if the iteration was not successful we decrease alpha and iterate with the same point
        if not successful:
            alpha = alpha / 2

        # generating the poll points which are on the mesh centered around oldpt
        x = np.random.randint(1, 5, size=d.shape) * alpha * d + oldpt[None, :]

    return plot, suc, pts


# can probably remove Rosenbrck and optimization code
def objective(trial):
    from more.gauss_full_cov import GaussFullCov
    from more.quad_model import QuadModelSubBLR
    from more.more_algo import MORE
    from more.sample_db import SimpleSampleDatabase
    import logging

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('MORE')
    logger.setLevel("INFO")

    dim = 15
    max_iters = 8  # multiply by samples_per_iter to get number of function evaluations

    # Int parameters
    samples_per_iter = 34  # trial.suggest_int('samples_per_iter', 15, 50)
    max_samples = int(samples_per_iter * 1.5)

    # Discrete-uniform parameters
    kl_bound = 2.6
    gamma = 0
    entropy_loss_bound = 3.0  # trial.suggest_discrete_uniform('entropy_loss_bound', 1.7, 4.1, 0.1)
    init_sigma = 3.3

    model_options_sub = {"normalize_features": True,
                         }

    more_config = {"epsilon": kl_bound,
                   "gamma": gamma,
                   "beta_0": entropy_loss_bound}

    # x_start = 0.5 * np.random.randn(dim)
    x_start = np.zeros(dim)
    # generates random starting point in the range -2 to 2
    for c in range(dim):
        x_start[c] = random.uniform(-2, 2)

    objective = BallInACupCostFunction(0)
    objective.newsuc()

    new_rewards = []
    rew = []
    count = 0  # stores the number of function evaluations
    avg = np.zeros(samples_per_iter)
    k = 0
    plot = []

    sample_db = SimpleSampleDatabase(max_samples)

    search_dist = GaussFullCov(x_start, init_sigma * np.eye(dim))
    surrogate = QuadModelSubBLR(dim, model_options_sub)

    more = MORE(dim, more_config, logger=logger)

    for i in range(max_iters):
        # creates samples based on search dist's mean + chol_cov * randomly generated vectors
        new_samples = search_dist.sample(samples_per_iter)  # the points to be optimized

        # new_rewards = objective(new_samples)
        # i can only pass 1 coordinate at a time to bicfun so i changed the above line to a loop
        for j in range(len(new_samples)):
            temp = objective(new_samples[j])
            new_rewards.append(-temp)  # negate the -ve reward, since MORE maximizes
            rew.append(temp)
            count += 1

            # reports reward and number of function evaluation that generated that reward
            trial.report(temp, count)

            # Handle pruning based on temp
            if trial.should_prune():
                raise optuna.TrialPruned()

            # taking the reward of the last iteration in avg
            if i == max_iters - 1:
                avg[k] = temp
                k += 1

        # adds new samples and rewards to the db, appending them to the current list
        sample_db.add_data(new_samples, new_rewards)

        # clearing the rewards
        new_rewards.clear()

        # setting samples and rewards to be the ones in database
        samples, rewards = sample_db.get_data()

        # success always seems to be true
        success = surrogate.fit(samples, rewards, search_dist, )
        if not success:
            continue

        new_mean, new_cov, success = more.step(search_dist, surrogate)

        if success:
            try:
                # updates search_dist's mean and cov to be the new ones
                search_dist.update_params(new_mean, new_cov)
            except Exception as e:
                print(e)

        # printing and saving the values from the iteration
        plot.append(rew)

    # taking the average of the last iteration
    print(np.mean(avg))
    return np.mean(avg)


def coordinate(a, dim, d, noiter, noruns, plot, suclist, funeval):
    # doing noruns runs and storing the values in plot, suclist and funeval
    for i in range(noruns):
        bic_fun = BallInACupCostFunction(0)

        # random starting point in range -2 to 2
        for c in range(dim):
            a[c] = random.uniform(-2, 2)

        tempplt, tempsuc = coordinate_search(bic_fun, a, d, noiter)
        plot.append(tempplt)
        funeval.append(bic_fun.retcount())
        suclist.append(tempsuc)

        print(i)
        print(tempsuc)
        print(bic_fun.retcount())  # recount should equal noiter
        print(tempplt)

    # creating a file per list of values to be stored and printing them
    np.save('successratecoordinate.npy', suclist)
    np.save('functionevaluationscoordinate.npy', funeval)
    np.save('rewardscoordinate.npy', plot)

    # printing the final list of all 3 values
    print(funeval)
    print(plot)
    print(suclist)


def direct(n, dim, d, noiter, noruns, plot, suclist, funeval, points):
    # n is the number of points, and b is the array to store these points, each having dimension dim
    b = np.zeros((n, dim))

    # doing noruns runs and storing the values
    for i in range(noruns):
        bic_fun = BallInACupCostFunction(0)

        # generating random starting points in the -2 to 2 range
        # n is the number of points in the array and d is their dimension
        for e in range(n):
            for c in range(dim):
                b[e][c] = random.uniform(-2, 2)

        tempplt, tempsuc, temppts = directional_direct_search(bic_fun, b, d, noiter)
        plot.append(tempplt)
        funeval.append(bic_fun.retcount())
        suclist.append(tempsuc.copy())
        points.append(temppts)

        print(i)
        print(tempsuc)
        print(bic_fun.retcount())  # retcount should equal noiter
        print(tempplt)
        print(temppts)

    # creating a file per list of values to be stored
    np.save('150successratedirectional.npy', suclist)
    np.save('150functionevaluationsdirectional.npy', funeval)
    np.save('150rewardsdirectional.npy', plot)
    np.save('150pointsdirectional.npy', points)

    # printing the final list of all 3 values
    print(funeval)
    print(plot)
    print(suclist)
    print(points)


# adapted from https://github.com/maxhuettenrauch/MORE
def MORE(dim, noiter, noruns, plot, suclist, funeval):
    from more.gauss_full_cov import GaussFullCov
    from more.quad_model import QuadModelSubBLR
    from more.more_algo import MORE
    from more.sample_db import SimpleSampleDatabase
    import logging

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('MORE')
    logger.setLevel("INFO")

    minimize = True
    samples_per_iter = 34  # you don't want this to be too big, it needs to be at least = dim
    max_samples = int(samples_per_iter * 1.5)

    # divide noiter by samples_per_iter to get max number of function evaluations
    max_iters = round(noiter/samples_per_iter)
    kl_bound = 2.6  # how much it moves to the next distribution
    gamma = 0  # prevents it from shrinking to fast, set to 0 to use the entropy instead
    entropy_loss_bound = 3.0  # also prevents shrinking
    init_sigma = 3.3

    model_options_sub = {"normalize_features": True,
                         }

    more_config = {"epsilon": kl_bound,
                   "gamma": gamma,
                   "beta_0": entropy_loss_bound}

    x_start = np.zeros(dim)

    # executes MORE for noruns runs
    for rep in range(noruns):

        objective = BallInACupCostFunction(0)
        objective.newsuc()

        # generates random starting point in the range -2 to 2
        for c in range(dim):
            x_start[c] = random.uniform(-2, 2)

        new_rewards = []
        rew = []
        count = 0  # stores the number of function evaluations

        sample_db = SimpleSampleDatabase(max_samples)

        search_dist = GaussFullCov(x_start, init_sigma * np.eye(dim))
        surrogate = QuadModelSubBLR(dim, model_options_sub)

        more = MORE(dim, more_config, logger=logger)

        for i in range(max_iters):
            logger.info("Iteration {}".format(i))

            # the points to be optimized
            new_samples = search_dist.sample(samples_per_iter)

            # new_rewards = objective(new_samples)
            # i can only pass 1 coordinate at a time to bicfun so i changed the above line to a loop
            for j in range(len(new_samples)):
                temp = objective(new_samples[j])
                new_rewards.append(-temp)  # negate the -ve reward, since MORE maximizes
                rew.append(temp)
                print(temp)
                count += 1

            # adds new samples and rewards to the db, appending them to the current list
            sample_db.add_data(new_samples, new_rewards)

            # clearing the rewards
            new_rewards.clear()

            # setting samples and rewards to be the ones in database
            samples, rewards = sample_db.get_data()

            # success always seems to be true
            success = surrogate.fit(samples, rewards, search_dist, )
            if not success:
                continue

            # updates some constants, setting self's old dist to search dist and self's current model to surrogate
            # success occurs if both kl and entropy are successful see dual_opt in more_algo
            new_mean, new_cov, success = more.step(search_dist, surrogate)

            if success:
                try:
                    # updates search_dist's mean and cov to be the new ones
                    search_dist.update_params(new_mean, new_cov)
                except Exception as e:
                    print(e)

            # printing and saving the values from the iteration
            plot.append(rew)
            funeval.append(count) # count should equal max_iters
            suclist.append(objective.retsuc())

    # creating a file per list of values to be stored and printing them
    np.save('successrateMORE.npy', suclist)
    np.save('functionevaluationsMORE.npy', funeval)
    np.save('rewardsMORE.npy', plot)
    print(funeval)
    print(plot)
    print(suclist)


# taken from https://www.zhangzk.net/software.html
def NEWUOA(dim, noiter, noruns, plot, suclist, funeval):
    # executes Powell NEWUOA for noruns runs
    for rep in range(noruns):

        # generates random starting point in the range -2 to 2
        for c in range(dim):
            a[c] = random.uniform(-2, 2)

        bic_fun = BallInACupCostFunction(0)

        # returns nfev, the no of fun evals, and fhist which holds the rewards
        res = newuoa(bic_fun, a, options={'maxfev': noiter})

        print(rep)
        print(bic_fun.retsuc())
        print(res.nfev)
        print(res.fhist)

        # storing the important values
        suclist.append(bic_fun.retsuc())
        funeval.append(res.nfev)
        plot.append(res.fhist)

        # clears the success list, in order not to repeat the list when saving it
        bic_fun.newsuc()

    plotnew = np.zeros((noruns, noiter))
    suclistnew = np.zeros((noruns, noiter))

    # since NEWUOA stops once it reaches a certain point we arrange suclist and plot to have noiter entries per run
    # in order to be able to compare it to the others, by repeating the final value
    for i in range(noruns):
        # get length of the current array
        ln = len(plot[i])

        # get the last value stored in the current array
        last = plot[i][ln - 1]

        # copying the entire array
        for j in range(ln):
            plotnew[i][j] = plot[i][j]

        # filling the rest of the array with the last value
        for j in range(ln, noiter):
            plotnew[i][j] = last

    for i in range(noruns):
        # get length of the current array
        ln = len(suclist[i])

        # get the last value stored in the current array
        last = suclist[i][ln - 1]

        # copying the entire array
        for j in range(ln):
            suclistnew[i][j] = suclist[i][j]

        # filling the rest of the array with the last value
        for j in range(ln, noiter):
            suclistnew[i][j] = last

    # creating a file per list of values to be stored
    np.save('successrateNEWUOA.npy', suclistnew)
    np.save('functionevaluationsNEWUOA.npy', funeval)
    np.save('rewardsNEWUOA.npy', plotnew)

    print(funeval)
    print(plotnew)
    print(suclistnew)


# taken from https://facebookresearch.github.io/nevergrad/optimizers_ref.html#optimizers (at very bottom of page)
def cGA(dim, noiter, noruns, plot, suclist, funeval):

    # executes GA for noruns runs
    for rep in range(noruns):

        bic_fun = BallInACupCostFunction(0)
        bic_fun.newrew()
        bic_fun.newsuc()

        # calling optimizer to minimize bic_fun, having dimensions dim and max iterations noiter
        nevergrad.optimization.optimizerlib.cGA(dim, noiter).minimize(bic_fun)

        print(rep)
        print(bic_fun.retreward())
        print(bic_fun.retsuc())

        funeval.append(noiter)
        plot.append(bic_fun.retreward())
        suclist.append(bic_fun.retsuc())

    # creating a file per list of values to be stored and printing them
    np.save('successrateGA.npy', suclist)
    np.save('functionevaluationsGA.npy', funeval)
    np.save('rewardsGA.npy', plot)

    print(funeval)
    print(plot)
    print(suclist)


# adapted from https://github.com/CMA-ES/pycma
def CMA(b, dim, noiter, noruns, plot, suclist, funeval):
    # executes CMA-ES for noruns runs
    for rep in range(noruns):

        # generates random starting point in the range -2 to 2
        for c in range(dim):
            b[c] = random.uniform(-2, 2)

        bic_fun = BallInACupCostFunction(0)
        bic_fun.newsuc()

        # Note number of iterations works in multiples of 12, so divide desired noiter by 12 and round
        maxi = round(noiter/12)
        alg = cma.CMAEvolutionStrategy(b, 1).optimize(bic_fun, maxi, verb_disp=1)

        # adapted from pascal's rep line 203 https://github.com/psclklnk/self-paced-rl/blob/master/run_experiment.py
        count = 0
        r = []
        s = []
        i = 0

        while count < maxi:
            thetas = alg.ask()  # gets a new list of solutions

            for theta in thetas:
                tmp = bic_fun(theta)
                r.append(tmp[0])
                s.append(tmp[1])

            count += len(thetas)  # number of function evaluations
            print(i)

            print("Count: %d" % count)

            # to have only last 12 rewards corresponding to the 12 thetas passed in alg.tell
            tempr = np.zeros(12)
            for j in range(12):
                tempr[j] = r[12 * i + j]

            alg.tell(thetas, tempr)  # pass objective function values to prepare for next iteration.
            alg.disp()  # print current state variables in a single-line.
            i += 1

        print(r)
        print(s)
        print(count)  # should be equal to maxi

        funeval.append(count)
        plot.append(r)
        suclist.append(s)

    # creating a file per list of values to be stored and printing them
    np.save('successrateCMA.npy', suclist)
    np.save('functionevaluationsCMA.npy', funeval)
    np.save('rewardsCMA.npy', plot)
    print(funeval)
    print(plot)
    print(suclist)


# main method
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    # setting number of dimensions
    dim = 15
    a = np.zeros(dim)
    plot = []
    suclist = []
    funeval = []
    points = []

    # generating the maximal basis for the given number of dimensions
    d = np.zeros((dim * 2, dim))
    for p in range(dim):
        d[p][p] = 1

    for p in range(dim):
        d[p + dim][p] = -1

    selection = ""

    # displaying the menu until the user chooses to quit
    while selection != "8":
        print("Menu of optimization algorithms")
        print("1. Coordinate Search")
        print("2. Direct Directional Search")
        print("3. MORE")
        print("4. Powell's NEWUOA")
        print("5. Nevergrad's cGA")
        print("6. CMA-ES")
        print("7. Plot reward and success against number of function evaluations")
        print("8. Quit")

        selection = input("Please select an option: ")

        # running the selected algorithm based on above selection
        # note that all the algorithms below will create files for the 3 lists generated, and display them at the end
        if selection == "1":
            noruns = int(input("Please enter how many runs you want to run: "))
            noiter = int(input("Please enter how many iterations you want to run: "))
            coordinate(a, dim, d, noiter, noruns, plot, suclist, funeval)

        elif selection == "2":
            noruns = int(input("Please enter how many runs you want to run: "))
            noiter = int(input("Please enter how many iterations you want to run: "))
            nopts = int(input("Please enter how many points you want to generate per iteration: "))
            direct(nopts, dim, d, noiter, noruns, plot, suclist, funeval, points)

        elif selection == "3":
            noruns = int(input("Please enter how many runs you want to run: "))
            noiter = int(input("Please enter how many iterations you want to run: "))
            MORE(dim, noiter,  noruns, plot, suclist, funeval)

        elif selection == "4":
            noruns = int(input("Please enter how many runs you want to run: "))
            noiter = int(input("Please enter how many iterations you want to run: "))
            NEWUOA(dim, noiter, noruns, plot, suclist, funeval)

        elif selection == "5":
            noruns = int(input("Please enter how many runs you want to run: "))
            noiter = int(input("Please enter how many iterations you want to run: "))
            cGA(dim, noiter, noruns, plot, suclist, funeval)

        elif selection == "6":
            noruns = int(input("Please enter how many runs you want to run: "))
            noiter = int(input("Please enter how many iterations you want to run: "))
            CMA(a, dim, noiter, noruns, plot, suclist, funeval)

        elif selection == "7":
            # if you want to plot given data, use noiter = 500
            noiter = int(input("Please enter how many iterations you have run for the algorithms: "))

            # note that if that if you change the below variables, you must also change them in the MORE code
            samples_per_iter = 34

            # calculating the number of iterations for the algorithms MORE and CMA-ES respectively
            max_iters = round(noiter / samples_per_iter)
            maxi = round(noiter / 12)

            # loading the contents from the files and storing them in the respective list to be used in plots
            suclistcoor = np.load('successratecoordinate.npy')
            funevalcoor = np.load('functionevaluationscoordinate.npy')
            plotcoor = np.load('rewardscoordinate.npy')

            suclistdir = np.load('successratedirectional.npy')
            funevaldir = np.load('functionevaluationsdirectional.npy')
            plotdir = np.load('rewardsdirectional.npy')

            suclistCMA = np.load('successrateCMA.npy')
            funevalCMA = np.load('functionevaluationsCMA.npy')
            plotCMA = np.load('rewardsCMA.npy')

            suclistGA = np.load('successrateGA.npy')
            funevalGA = np.load('functionevaluationsGA.npy')
            plotGA = np.load('rewardsGA.npy')

            # has diff lengths but repeated last entry earlier to make them all have a uniform size of noiter
            suclistNEWUOA = np.load('successrateNEWUOA.npy')
            funevalNEWUOA = np.load('functionevaluationsNEWUOA.npy')
            plotNEWUOA = np.load('rewardsNEWUOA.npy')

            suclistMORE = np.load('successrateMORE.npy')
            funevalMORE = np.load('functionevaluationsMORE.npy')
            plotMORE = np.load('rewardsMORE.npy')

            # plotting number of function evaluations
            # coordinate search

            # taking the average of the function values
            perc = np.squeeze(plotcoor).T
            per5 = np.quantile(perc, 0.05, axis=1)
            avgplot = np.mean(perc, axis=1)
            per95 = np.quantile(perc, 0.95, axis=1)

            # plotting the average of noruns runs of coordinate search with blue dashes
            plt.plot(per5, 'b--')  # , label="5th percentile"
            plt.plot(avgplot, 'b-', label="coordinate search")  # average
            plt.plot(per95, 'b--')  # , label="95th percentile"

            # doing in between percentile shading
            x = np.arange(0, noiter, 1)
            plt.fill_between(x, per5, per95, facecolor='blue', alpha=0.5)

            # direct directional search

            perc = np.squeeze(plotdir).T
            per5 = np.quantile(perc, 0.05, axis=1)
            avgplot = np.mean(perc, axis=1)
            per95 = np.quantile(perc, 0.95, axis=1)

            # plotting the average, 5th and 95th percentile of direct search in red
            plt.plot(per5, 'r--')  # , label="5th percentile"
            plt.plot(avgplot, 'r-', label="direct search")  # average
            plt.plot(per95, 'r--')  # , label="95th percentile"

            # doing in between percentile shading
            x = np.arange(0, noiter, 1)
            plt.fill_between(x, per5, per95, facecolor='red', alpha=0.5)

            # nevergrad's cGA

            # taking the average of the function values
            perc = np.squeeze(plotGA).T
            per5 = np.quantile(perc, 0.05, axis=1)
            avgplot = np.mean(perc, axis=1)
            per95 = np.quantile(perc, 0.95, axis=1)

            # plotting the average of noruns runs of GA with green dashes
            plt.plot(per5, 'g--')  # , label="5th percentile"
            plt.plot(avgplot, 'g-', label="Genetic Algorithm")  # average
            plt.plot(per95, 'g--')  # , label="95th percentile"

            # doing in between percentile shading
            x = np.arange(0, noiter, 1)
            plt.fill_between(x, per5, per95, facecolor='green', alpha=0.5)

            # CMA-ES

            perc = np.squeeze(plotCMA).T
            per5 = np.quantile(perc, 0.05, axis=1)
            avgplot = np.mean(perc, axis=1)
            per95 = np.quantile(perc, 0.95, axis=1)

            # plotting the average, 5th and 95th percentile of CMA-ES in cyan
            plt.plot(per5, 'c--')  # , label="5th percentile"
            plt.plot(avgplot, 'c-', label="CMA-ES")  # average
            plt.plot(per95, 'c--')  # , label="95th percentile"

            # doing in between percentile shading
            x = np.arange(0, maxi, 1)
            plt.fill_between(x, per5, per95, facecolor='cyan', alpha=0.5)

            # MORE

            # taking the average of the function values
            perc = np.squeeze(plotMORE).T
            per5 = np.quantile(perc, 0.05, axis=1)
            avgplot = np.mean(perc, axis=1)
            per95 = np.quantile(perc, 0.95, axis=1)

            # plotting the average of noruns runs of MORE with magenta dashes
            plt.plot(per5, 'm--')  # , label="5th percentile"
            plt.plot(avgplot, 'm-', label="MORE")  # average
            plt.plot(per95, 'm--')  # , label="95th percentile"

            # doing in between percentile shading
            x = np.arange(0, max_iters, 1)
            plt.fill_between(x, per5, per95, facecolor='magenta', alpha=0.5)

            # Powell's NEWUOA

            perc = np.squeeze(plotNEWUOA).T
            per5 = np.quantile(perc, 0.05, axis=1)
            avgplot = np.mean(perc, axis=1)
            per95 = np.quantile(perc, 0.95, axis=1)

            # plotting the average, 5th and 95th percentile of NEWUOA in yellow
            plt.plot(per5, 'y--')  # , label="5th percentile"
            plt.plot(avgplot, 'y-', label="NEWUOA")  # average
            plt.plot(per95, 'y--')  # , label="95th percentile"

            # doing in between percentile shading
            x = np.arange(0, noiter, 1)
            plt.fill_between(x, per5, per95, facecolor='yellow', alpha=0.5)

            # setting the axes
            plt.xlim(0, noiter)
            plt.ylim(-1, 0)

            # labelling the axes and displaying the plot
            plt.ylabel('Reward')
            plt.xlabel('Number of function evaluations')
            plt.legend()
            plt.show()
            plt.close()

            # plotting the success rates of the algorithms
            # coordinate search

            # taking the average of success
            perc = np.squeeze(suclistcoor).T
            per5 = np.quantile(perc, 0.05, axis=1)
            avgplot = np.mean(perc, axis=1)
            per95 = np.quantile(perc, 0.95, axis=1)

            # plotting the average of noruns runs of coordinate search with red dashes
            plt.plot(per5, 'b--')  # , label="5th percentile"
            plt.plot(avgplot, 'b-', label="coordinate search")  # average
            plt.plot(per95, 'b--')  # , label="95th percentile"

            # direct directional search

            # taking the average of noruns runs of direct search
            perc2 = np.squeeze(suclistdir).T
            per5 = np.quantile(perc2, 0.05, axis=1)
            avgplot = np.mean(perc2, axis=1)
            per95 = np.quantile(perc2, 0.95, axis=1)

            # plotting the average of the values with red
            plt.plot(per5, 'r--')  # , label="5th percentile"
            plt.plot(avgplot, 'r-', label="direct directional search")  # average
            plt.plot(per95, 'r--')  # , label="95th percentile"

            # nevergrad's cGA

            # taking the average of success
            perc = np.squeeze(suclistGA).T
            per5 = np.quantile(perc, 0.05, axis=1)
            avgplot = np.mean(perc, axis=1)
            per95 = np.quantile(perc, 0.95, axis=1)

            # plotting the average of noruns runs of GA with green dashes
            plt.plot(per5, 'g--')  # , label="5th percentile"
            plt.plot(avgplot, 'g-', label="Genetic Algorithm")  # average
            plt.plot(per95, 'g--')  # , label="95th percentile"

            # CMA-ES

            # taking the average of noruns runs of CMA-ES
            perc2 = np.squeeze(suclistCMA).T
            per5 = np.quantile(perc2, 0.05, axis=1)
            avgplot = np.mean(perc2, axis=1)
            per95 = np.quantile(perc2, 0.95, axis=1)

            # plotting the average of the values with cyan
            plt.plot(per5, 'c--')  # , label="5th percentile"
            plt.plot(avgplot, 'c-', label="CMA-ES")  # average
            plt.plot(per95, 'c--')  # , label="95th percentile"

            # MORE

            # taking the average of success
            perc = np.squeeze(suclistMORE).T
            per5 = np.quantile(perc, 0.05, axis=1)
            avgplot = np.mean(perc, axis=1)
            per95 = np.quantile(perc, 0.95, axis=1)

            # plotting the average of noruns runs of MORE with magenta dashes
            plt.plot(per5, 'm--')  # , label="5th percentile"
            plt.plot(avgplot, 'm-', label="MORE")  # average
            plt.plot(per95, 'm--')  # , label="95th percentile"

            # Powell's NEWUOA

            # taking the average of noruns runs of NEWUOA
            perc2 = np.squeeze(suclistNEWUOA).T
            per5 = np.quantile(perc2, 0.05, axis=1)
            avgplot = np.mean(perc2, axis=1)
            per95 = np.quantile(perc2, 0.95, axis=1)

            # plotting the average of the values with yellow
            plt.plot(per5, 'y--')  # , label="5th percentile"
            plt.plot(avgplot, 'y-', label="NEWUOA")  # average
            plt.plot(per95, 'y--')  # , label="95th percentile"

            # setting the axes
            plt.xlim(0, noiter)
            plt.ylim(0, 1)

            # labelling the axes and showing the graph
            plt.ylabel('Success rate')
            plt.xlabel('Number of function evaluations')
            plt.legend()
            plt.show()
            plt.close()

        elif selection == "8":
            print("Quitting")
        else:  # prints error message if none of the options are selected
            print("Invalid input. Please try again.")

    """    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=50, interval_steps=20))
    study.optimize(objective, n_trials=50)
    print(len(study.trials))
    print(study.best_params)"""



