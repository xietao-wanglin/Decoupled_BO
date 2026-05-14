import os
import pickle


class Results:
    def __init__(self, filename):
        self.filename = filename
        self.input_data = None
        self.output_data = None
        self.seed = None
        self.performance_type = None
        self.number_initial_samples = None
        self.budget = None
        self.cost_configurations = None
        self.budget_consumed = []
        self.model_length_scales = []
        self.acqf_values = []
        self.best_predicted_location = []
        self.best_predicted_location_true_value = []
        self.acqf_recommended_location = []
        self.acqf_recommended_location_value = []
        self.acqf_recommended_output_index = []
        self.failing_constraint = []
        self.evals = []

    def save_budget_consumed(self, budget_consumed):
        self.budget_consumed.append(budget_consumed.item())

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

    def save_acqf_values(self, acqf_values):
        self.acqf_values.append(acqf_values)

    def save_model_length_scales(self, model_length_scales):
        self.model_length_scales.append(model_length_scales)

    def save_cost_configurations(self, cost_configuration):
        self.cost_configurations = cost_configuration

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

    @classmethod
    def load_from_file(cls, filepath):
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        r = cls(filename=d['filename'])
        r.filepath = d['path']
        r.input_data = d['input_data']
        r.output_data = d['output_data']
        r.seed = d['seed']
        r.performance_type = d['performance_type']
        r.number_initial_samples = d['number_initial_designs']
        r.budget = d['budget']
        r.cost_configurations = d['cost_configurations']
        r.budget_consumed = list(d['budget_consumed'])
        r.model_length_scales = list(d['model_lengthscales'])
        r.acqf_values = list(d['acqf_values'])
        r.best_predicted_location = list(d['best_predicted_location'])
        r.best_predicted_location_true_value = list(d['best_predicted_location_value'])
        r.acqf_recommended_location = list(d['acqf_recommended_location'])
        r.acqf_recommended_location_value = list(d['acqf_recommended_location_value'])
        r.acqf_recommended_output_index = list(d['acqf_recommended_output_index:'])
        r.failing_constraint = list(d['failing_index:'])
        r.evals = list(d['evaluated_functions'])
        return r

    def _build_results_dict(self):
        return {"path": self.filepath,
                "filename": self.filename,
                "model_lengthscales": self.model_length_scales,
                "number_initial_designs": self.number_initial_samples,
                "budget": self.budget,
                "cost_configurations": self.cost_configurations,
                "performance_type": self.performance_type,
                "seed": self.seed,
                "input_data": self.input_data,
                "output_data": self.output_data,
                "best_predicted_location": self.best_predicted_location,
                "best_predicted_location_value": self.best_predicted_location_true_value,
                "acqf_recommended_location": self.acqf_recommended_location,
                "acqf_recommended_location_value": self.acqf_recommended_location_value,
                "acqf_recommended_output_index:": self.acqf_recommended_output_index,
                "acqf_values": self.acqf_values,
                "failing_index:": self.failing_constraint,
                "evaluated_functions": self.evals,
                "budget_consumed": self.budget_consumed}
