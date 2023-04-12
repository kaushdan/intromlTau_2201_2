#################################
# Your name: Daniel Kaushansky
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        rng = np.random.default_rng()
        xs = np.sort(rng.uniform(size=m))
        ys = np.array(
            [
                rng.choice(
                    [0, 1],
                    p=[
                        self._y_given_x_prob(x=x, y=0),
                        self._y_given_x_prob(x=x, y=1),
                    ],
                )
                for x in xs
            ]
        )

        return np.array(list(zip(xs, ys)))

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        print("In experiment_m_range_erm")
        outcomes = []

        for m in range(m_first, m_last + 1, step):
            true_err_sum = emp_err_sum = 0
            for _ in range(T):
                sample = self.sample_from_D(m)
                erm_h, erm_emp_err = intervals.find_best_interval(
                    xs=sample[:, 0], ys=sample[:, 1], k=k
                )
                emp_err_sum += erm_emp_err
                true_err_sum += self._calculate_true_error(erm_h)

            emp_err_avg = emp_err_sum / (T * m)
            true_err_avg = true_err_sum / T
            outcomes.append((emp_err_avg, true_err_avg))

        outcomes = np.array(outcomes)
        self._scatter_experiment(
            xs=np.arange(m_first, m_last + 1, step),
            outcomes=outcomes,
            suptitle="experiment_m_range_erm",
            ax1_title="Averaged empirical error as function of sample size",
            ax1_x_label="sample size",
            ax1_y_label="Averaged empirical error",
            ax2_title="Averaged true error as function of sample size",
            ax2_x_label="sample size",
            ax2_y_label="Averaged true error",
        )

        return np.array(outcomes)

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        print("In experiment_k_range_erm")

        outcomes = []
        sample = self.sample_from_D(m)

        for k in range(k_first, k_last + 1, step):
            erm_h, erm_emp_err = intervals.find_best_interval(
                xs=sample[:, 0], ys=sample[:, 1], k=k
            )
            erm_true_err = self._calculate_true_error(erm_h)
            outcomes.append((erm_emp_err / m, erm_true_err))

        outcomes = np.array(outcomes)
        self._scatter_experiment(
            xs=np.arange(k_first, k_last + 1, step),
            outcomes=outcomes,
            suptitle="experiment_k_range_erm",
            ax1_title="Empirical error as function of k",
            ax1_x_label="k",
            ax1_y_label="Empirical error",
            ax2_title="True error as function of sample size",
            ax2_x_label="k",
            ax2_y_label="True error",
        )

        emp_error = outcomes[:, 0]
        min_k = np.argmin(emp_error) + 1
        print(f"The k that minimizes erm empirical error is: {min_k}")

        return min_k

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        print("In experiment_k_range_srm")

        outcomes = []
        sample = self.sample_from_D(m)
        ks = np.arange(k_first, k_last + 1, step)

        for k in ks:
            erm_h, erm_emp_err = intervals.find_best_interval(
                xs=sample[:, 0], ys=sample[:, 1], k=k
            )
            erm_true_err = self._calculate_true_error(erm_h)
            outcomes.append((erm_emp_err / m, erm_true_err))

        outcomes = np.array(outcomes)

        self._plot_d_experiment(ks=ks, outcomes=outcomes, m=m, delta=0.1)

        srm_results = outcomes[:, 0] + self._penalty_function(
            ks, m=m, delta=0.1
        )

        min_k = np.argmin(srm_results) + 1
        print(f"The k that minimizes srm is: {min_k}")

        return min_k

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        print("In cross_validation")

        sample = self.sample_from_D(m)
        ks = np.arange(1, 10 + 1, 1)

        rng = np.random.default_rng()
        idx = rng.choice(m, (m * 8) // 10, replace=False)
        s_test = sample[idx]
        s_test = s_test[s_test[:, 0].argsort()]
        mask = np.ones(m, dtype=bool)
        mask[idx] = False
        s_holdout = sample[mask]

        hs = np.array(
            [
                intervals.find_best_interval(
                    xs=s_test[:, 0], ys=s_test[:, 1], k=k
                )
                for k in ks
            ],
            dtype=object,
        )

        s_holdout_err = [
            self._calculate_empirical_err(h, sample=s_holdout) for h, _ in hs
        ]
        min_k = np.argmin(s_holdout_err) + 1

        self._plot_cross_validation(
            ks=ks, test_err=hs[:, 1] / m, holdout_err=s_holdout_err
        )

        print(f"The best k according to holout-validation is: {min_k}")

        return min_k

    #################################
    # Place for additional methods

    def _scatter_experiment(
        self,
        xs,
        outcomes,
        suptitle,
        ax1_title,
        ax1_x_label,
        ax1_y_label,
        ax2_title,
        ax2_x_label,
        ax2_y_label,
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(suptitle)
        ax1.set_title(ax1_title)
        ax1.set_xlabel(ax1_x_label)
        ax1.set_ylabel(ax1_y_label)
        ax2.set_title(ax2_title)
        ax2.set_xlabel(ax2_x_label)
        ax2.set_ylabel(ax2_y_label)

        for (avg_emp_err, avg_true_err), x in zip(outcomes, xs):
            ax1.scatter(x, avg_emp_err)
            ax2.scatter(x, avg_true_err)

        plt.show(block=False)

    def _plot_cross_validation(self, ks, test_err, holdout_err):
        plt.plot(ks, test_err, "g", label="Empirical error over test")
        plt.plot(ks, holdout_err, "r", label="Empirical error over validation")
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

    def _plot_d_experiment(self, ks, outcomes, m, delta):
        plt.plot(ks, outcomes[:, 0], "g", label="Empirical error")
        plt.plot(ks, outcomes[:, 1], "r", label="True error")
        plt.plot(
            ks,
            self._penalty_function(ks, m=m, delta=delta),
            "m",
            label="Penalty function",
        )
        plt.plot(
            ks,
            outcomes[:, 0] + self._penalty_function(ks),
            "c",
            label="Sum of penalty and erm empirical error",
        )
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

    def _penalty_function(self, k, m=1500, delta=0.1):
        return 2 * np.sqrt((2 * k + np.log(2 * len(k) / delta)) / m)

    def _calculate_empirical_err(self, h, sample):
        return sum(self._apply_h_on_x(h, x) != y for x, y in sample) / len(
            sample
        )

    def _apply_h_on_x(self, h, x):
        for interval_start, interval_end in h:
            if interval_start <= x <= interval_end:
                return 1

        return 0

    def _calculate_true_error(self, h) -> float:
        """Return the true error of h

        Given a hyphothesis in H_k this method calculated it's true error
        using the distibution implemeted in _y_given_x.

        Args:
            h (sequence): hyphothesis in H_k.

        Returns:
            float. The true error of h.
        """
        return self._false_negative_prob(h) + self._false_positive_prob(h)

    def _false_positive_prob(self, h) -> float:
        """Calculate the probability of false positive occurences.

        This method calculates P[{(x,0)∈Ω : h(x)=1}].

        Args:
            h (sequence): hyphothesis in H_k.

        Returns:
            float. P[{(x,0)∈Ω : h(x)=1}].
        """
        splitted_intervals = self._split_intervals(h)

        intervals = (
            splitted_interval
            for splitted_interval in splitted_intervals
            if any(
                self._is_sub_interval(splitted_interval, interval)
                for interval in h
            )
        )

        return self._calc_intervals_prob(intervals, y=0)

    def _is_sub_interval(self, first, second):
        """Returns if first is contained in second.

        Args:
            first (sequence): interval in R.
            second (seq uence): interval in R.

        Returns:
            bool. If first interval is contained in second interval.

        """
        f_start, f_end = first
        s_start, s_end = second

        return f_start >= s_start and f_end <= s_end

    def _false_negative_prob(self, h) -> float:
        """Calculate the probability of false positive occurences.

        This method calculates P[{(x,1)∈Ω : h(x)=0}].

        Args:
            h (sequence): hyphothesis in H_k.

        Returns:
            float. P[{(x,1)∈Ω : h(x)=0}].
        """
        splitted_intervals = self._split_intervals(h)

        intervals = (
            splitted_interval
            for splitted_interval in splitted_intervals
            if all(
                not self._is_sub_interval(splitted_interval, interval)
                for interval in h
            )
        )

        return self._calc_intervals_prob(intervals, y=1)

    def _calc_intervals_prob(self, intervals, y) -> float:
        return sum(
            (end - start) * self._y_given_x_prob(x=(start + end) / 2, y=y)
            for start, end in intervals
        )

    def _split_intervals(self, intervals):
        middle_points = (0, 0.2, 0.4, 0.6, 0.8, 1)
        flatten_intervals = tuple(
            [point for interval in intervals for point in interval]
        )

        new_locations = sorted(tuple(set(middle_points + flatten_intervals)))

        return [
            (new_locations[i], new_locations[i + 1])
            for i in range(len(new_locations) - 1)
        ]

    def _y_given_x_prob(self, y, x) -> float:
        def positive_label(x):
            if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
                return 0.8

            return 0.1

        return positive_label(x) if y == 1 else 1 - positive_label(x)

    #################################


if __name__ == "__main__":
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
