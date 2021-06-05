#!/usr/bin/env python3
# Copyright 2010-2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Creates a shift scheduling problem and solves it."""

from absl import app
from absl import flags

from ortools.sat.python import cp_model
from google.protobuf import text_format

FLAGS = flags.FLAGS

flags.DEFINE_string('output_proto', '',
                    'Output file to write the cp_model proto to.')
flags.DEFINE_string('params', 'max_time_in_seconds:10.0',
                    'Sat solver parameters.')


def negated_bounded_span(works, start, length):
    """Filters an isolated sub-sequence of variables assined to True.

  Extract the span of Boolean variables [start, start + length), negate them,
  and if there is variables to the left/right of this span, surround the span by
  them in non negated form.

  Args:
    works: a list of variables to extract the span from.
    start: the start to the span.
    length: the length of the span.

  Returns:
    a list of variables which conjunction will be false if the sub-list is
    assigned to True, and correctly bounded by variables assigned to False,
    or by the start or end of works.
  """
    sequence = []
    # Left border (start of works, or works[start - 1])
    if start > 0:
        sequence.append(works[start - 1])
    for i in range(length):
        sequence.append(works[start + i].Not())
    # Right border (end of works or works[start + length])
    if start + length < len(works):
        sequence.append(works[start + length])
    return sequence


def add_soft_sequence_constraint(model, works, hard_min, hard_max, prefix):
    """Sequence constraint on true variables with soft and hard bounds.

  This constraint look at every maximal contiguous sequence of variables
  assigned to true. If forbids sequence of length < hard_min or > hard_max.
  Then it creates penalty terms if the length is < soft_min or > soft_max.

  Args:
    model: the sequence constraint is built on this model.
    works: a list of Boolean variables.
    hard_min: any sequence of true variables must have a length of at least
      hard_min.
    soft_min: any sequence should have a length of at least soft_min, or a
      linear penalty on the delta will be added to the objective.
    min_cost: the coefficient of the linear penalty if the length is less than
      soft_min.
    soft_max: any sequence should have a length of at most soft_max, or a linear
      penalty on the delta will be added to the objective.
    hard_max: any sequence of true variables must have a length of at most
      hard_max.
    max_cost: the coefficient of the linear penalty if the length is more than
      soft_max.
    prefix: a base name for penalty literals.

  Returns:
    a tuple (variables_list, coefficient_list) containing the different
    penalties created by the sequence constraint.
  """
    cost_literals = []
    cost_coefficients = []

    # Forbid sequences that are too short.
    for length in range(1, hard_min):
        for start in range(len(works) - length + 1):
            model.AddBoolOr(negated_bounded_span(works, start, length))

    # Just forbid any sequence of true variables with length hard_max + 1
    for start in range(len(works) - hard_max):
        model.AddBoolOr(
            [works[i].Not() for i in range(start, start + hard_max + 1)])
    return cost_literals, cost_coefficients


def add_soft_sum_constraint(model, works, hard_min, hard_max, prefix):
    """Sum constraint with soft and hard bounds.

  This constraint counts the variables assigned to true from works.
  If forbids sum < hard_min or > hard_max.
  Then it creates penalty terms if the sum is < soft_min or > soft_max.

  Args:
    model: the sequence constraint is built on this model.
    works: a list of Boolean variables.
    hard_min: any sequence of true variables must have a sum of at least
      hard_min.
    soft_min: any sequence should have a sum of at least soft_min, or a linear
      penalty on the delta will be added to the objective.
    min_cost: the coefficient of the linear penalty if the sum is less than
      soft_min.
    soft_max: any sequence should have a sum of at most soft_max, or a linear
      penalty on the delta will be added to the objective.
    hard_max: any sequence of true variables must have a sum of at most
      hard_max.
    max_cost: the coefficient of the linear penalty if the sum is more than
      soft_max.
    prefix: a base name for penalty variables.

  Returns:
    a tuple (variables_list, coefficient_list) containing the different
    penalties created by the sequence constraint.
  """
    cost_variables = []
    cost_coefficients = []
    sum_var = model.NewIntVar(hard_min, hard_max, '')
    # This adds the hard constraints on the sum.
    model.Add(sum_var == sum(works))

    return cost_variables, cost_coefficients


def solve_shift_scheduling(params, output_proto):
    """Solves the shift scheduling problem."""
    # Data
    num_employees = 16
    num_weeks = 2
    shifts = ['D', 'N']
    day_shift = 0
    night_shift = 1
	
    num_days = num_weeks * 7
    num_shifts = len(shifts)

    # The required number of shifts worked per week
    max_shifts_per_week_constraint = (0, 7)
	
	
	# The required number of day shifts in a 2 week schedule
    day_shifts_per_two_weeks = (1, num_days * num_shifts)
	
    # Number of nurses that MUST be assigned to each shift
    required_nurses_per_shift = 7

    model = cp_model.CpModel()

    work = {}
    for e in range(num_employees):
        for d in range(num_days):
            for s in range(num_shifts):
                work[e, d, s] = model.NewBoolVar('work%i_%i_%i' % (e, d, s))


    # Handle the required nurses per shift constraint
    for s in range(num_shifts):
        for w in range(num_weeks):
            for d in range(7):
                works = [work[e, w * 7 + d, s] for e in range(num_employees)]
                # Ignore Off shift.
                worked = model.NewIntVar(required_nurses_per_shift, required_nurses_per_shift, '')
                model.Add(worked == sum(works))
				
    # Handle the min day shifts per 2 weeks constraint
    hard_min, hard_max = day_shifts_per_two_weeks
    for e in range(num_employees):
        works = [work[e, d, day_shift] for d in range(num_days)]
        variables, coeffs = add_soft_sum_constraint(
                model, works, hard_min, hard_max,
                'weekly_sum_constraint(employee %i, day shift)' %
                (e))

    # Handle the max shifts per week constraint
    hard_min, hard_max = max_shifts_per_week_constraint
    for e in range(num_employees):
        for w in range(num_weeks):
            works = [work[e, d + w * 7, s] for d in range(7) for s in range(num_shifts)]
            variables, coeffs = add_soft_sum_constraint(
                model, works, hard_min, hard_max,
                'weekly_sum_constraint(employee %i, shift %i, week %i)' %
                (e, s, w))
				
			


    if output_proto:
        print('Writing proto to %s' % output_proto)
        with open(output_proto, 'w') as text_file:
            text_file.write(str(model))

    # Solve the model.
    solver = cp_model.CpSolver()
    if params:
        text_format.Parse(params, solver.parameters)
    solution_printer = cp_model.ObjectiveSolutionPrinter()
    status = solver.Solve(model, solution_printer)

    # Print solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print()
        header = '          '
        for w in range(num_weeks):
            header += ' M   T   W   T   F   S   S   '
        print(header)
        for e in range(num_employees):
            schedule = ''
            for d in range(num_days):
                schedule_to_add = ''
                for s in range(num_shifts):
                    if solver.BooleanValue(work[e, d, s]):
                        schedule_to_add += shifts[s]
                schedule += schedule_to_add.ljust(4)
            print('worker %2i: %s' % (e, schedule))
        print()

    print()
    print('Statistics')
    print('  - status          : %s' % solver.StatusName(status))
    print('  - conflicts       : %i' % solver.NumConflicts())
    print('  - branches        : %i' % solver.NumBranches())
    print('  - wall time       : %f s' % solver.WallTime())


def main(_):
    solve_shift_scheduling(FLAGS.params, FLAGS.output_proto)


if __name__ == '__main__':
    app.run(main)
