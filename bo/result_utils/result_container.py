import os
import pickle


class Results():
    def __init__(self, filename):
        self.filename = filename
        self.input_data = None
        self.output_data = None
        self.seed = None
        self.performance_type = None
        self.number_initial_samples = None
        self.budget = None
        self.acqf_values = []
        self.best_predicted_location = []
        self.best_predicted_location_true_value = []
        self.acqf_recommended_location = []
        self.acqf_recommended_location_value = []
        self.acqf_recommended_output_index = []
        self.failing_constraint = []
        self.evals = []

    def save_failing_constraint(self, k):
        if k == -1:
            self.failing_constraint.append("None")
        else:
            self.failing_constraint.append(k)

    def save_input_data(self, x):
        self.input_data = x

    def save_output_data(self, y):
        self.output_data = y

    def save_best_predicted_location(self, location):
        self.best_predicted_location.append(location)  # xr in cKG paper....

    def save_best_predicted_location_true_value(self, value):
        self.best_predicted_location_true_value.append(value)  # f(xr) in cKG paper

    def save_acqf_recommended_location(self, location):
        self.acqf_recommended_location.append(location)

    def save_acqf_recommended_location_true_value(self, value):
        self.acqf_recommended_location_value.append(value)

    def save_acqf_recommended_output_index(self, index):
        self.acqf_recommended_output_index.append(index)

    def save_performance_type(self, performance_type):
        self.performance_type = performance_type

    def save_number_initial_points(self, number_initial_designs):
        self.number_initial_samples = number_initial_designs

    def random_seed(self, seed):
        self.seed = seed

    def save_budget(self, budget):
        self.budget = budget

    def save_evaluated_functions(self, evals):
        self.evals.append(evals)

    def generate_pkl_file(self):
        # Create a directory called 'results' if it doesn't exist
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Join the directory path and the filename
        self.filepath = os.path.join(results_dir, self.filename)

        results_dict = self._build_results_dict()

        # Serialize the results and save them to a pickle file
        with open(self.filepath, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f"Results saved to: {self.filepath} \n")

    def _build_results_dict(self):
        return {"path": self.filepath,
                "filename": self.filename,
                "number_initial_designs": self.number_initial_samples,
                "budget": self.budget,
                "performance_type": self.performance_type,
                "seed": self.seed,
                "input_data": self.input_data,
                "output_data": self.output_data,
                "best_predicted_location": self.best_predicted_location,
                "best_predicted_location_value": self.best_predicted_location_true_value,
                "acqf_recommended_location": self.acqf_recommended_location,
                "acqf_recommended_location_value": self.acqf_recommended_location_value,
                "acqf_recommended_output_index:": self.acqf_recommended_output_index,
                "failing_index:": self.failing_constraint,
                "evaluated_functions": self.evals}

    def save_acqf_values(self, acqf_values):
        self.acqf_values.append(acqf_values)
