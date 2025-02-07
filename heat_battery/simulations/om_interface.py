from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Model, fmi2False, POINTER, c_double
from fmpy.sundials import CVodeSolver
from fmpy.simulation import (
    Recorder,  settable_in_instantiated, settable_in_initialization_mode, 
    apply_start_values, fmi2Real, fmi2Integer, fmi2Boolean, c_uint32, c_int)
import numpy as np
from collections import defaultdict
import re
import pandas as pd


class InputSetterOMFMU2():
    def __init__(self, fmu, modelDescription, f_dict, events):
        self.fmu = fmu
        self.modelDescription = modelDescription
        self.f_dict = f_dict

        inputs = self.find_all_settable(modelDescription)
        continuous_inputs = self.lump_arrays(inputs, is_continuous=True)
        discrete_inputs = self.lump_arrays(inputs, is_continuous=False)
        self.continuous_setters = self.define_setters(continuous_inputs, f_dict)
        self.discrete_setters = self.define_setters(discrete_inputs, f_dict)
        self.nextEvent = self.define_nextEvent_callable(events)

    def apply_setters(self, time, setters):
        for all_refs, values, func, setter in setters:
            values[:] = np.concatenate([np.atleast_1d(f(time)).flatten() for f in func])
            setter(self.fmu.component, all_refs, len(all_refs), values)

    def apply(self, time, continuous=True, discrete=True, after_event=False):
        if after_event:
            time += 1e-13
        if continuous:
            self.apply_setters(time, self.continuous_setters)
        if discrete:
            self.apply_setters(time, self.discrete_setters)

    def find_all_settable(self, modelDescription):
        inputs = []
        for i, variable in enumerate(modelDescription.modelVariables):
            if variable.causality == 'input' or variable.variability == 'tunable':
                inputs.append(i)
        return inputs

    def lump_arrays(self, inputs, is_continuous=True):
        var_base = defaultdict(
            lambda : dict(
                type=None, causality=None, variability=None, 
                shape=0, full_names=[], refs=[], mdidxs=[]))
        for i in inputs:
            variable = self.modelDescription.modelVariables[i]
            is_real = variable.type in ['Float32', 'Float64', 'Real']
            is_discrete = variable.variability in ['discrete', 'tunable']
            _is_continuous = is_real and not is_discrete
            if _is_continuous==is_continuous:
                base_name, idxs = self.split_base_and_index(variable.name)
                d = var_base[base_name]
                d['type'] = variable.type
                d['causality'] = variable.causality
                d['variability'] = variable.variability
                d['shape'] = np.maximum(d['shape'], idxs)
                d['full_names'].append(variable.name)
                d['refs'].append(variable.valueReference)
                d['mdidxs'].append(i)
        return var_base
    
    def split_base_and_index(self, var_name):
        res = re.search('\[[0-9,]*\]$', var_name)
        if res is not None:
            start = res.start()
            bracket = res.group()
            base_name = var_name[:start]
            idxs = np.array(list(map(int, re.findall('\d+', bracket))))
            return base_name, idxs
        else:
            return var_name, np.ones(1, dtype=int)

    def define_setters(self, var_base, f_dict, ignore_missing_input=False, ignore_missing_tunable=False):
        set_methods_and_types = dict()
        set_methods_and_types['Real'] = (self.fmu.fmi2SetReal,    fmi2Real)
        set_methods_and_types['Integer'] = (self.fmu.fmi2SetInteger, fmi2Integer)
        set_methods_and_types['Boolean'] = (self.fmu.fmi2SetBoolean, fmi2Boolean)
        set_methods_and_types['Enumeration'] = (self.fmu.fmi2SetInteger, fmi2Integer)

        _type_base_refs = defaultdict(list)
        _type_base_func = defaultdict(list)

        setters = []
        for base_name, data in var_base.items():
            if not f_dict.get(base_name) and data['causality'] == 'input' and not ignore_missing_input:
                print(f'Warning: missing input for variable "{base_name}" of shape {data["shape"]}')
                continue
            if not f_dict.get(base_name) and data['variability'] == 'tunable' and not ignore_missing_tunable:
                print(f'Warning: missing tunable for variable "{base_name}" of shape {data["shape"]}')
                continue

            _type = data['type']
            _type_base_refs[_type] += data['refs']
            _type_base_func[_type].append(f_dict[base_name])

            e = f'Error: Shape {data["shape"]} of variable "{base_name}" does not match the return shape from the setter'
            assert len(np.atleast_1d(f_dict[base_name](0.0)).flatten()) == np.prod(data['shape']), e

        for _type, all_refs in _type_base_refs.items():
            setter, setter_type = set_methods_and_types[_type]
            setters.append((
                (c_uint32 * len(all_refs))(*all_refs),
                (setter_type * len(all_refs))(),
                _type_base_func[_type],
                setter
            ))
        return setters
    
    def define_nextEvent_callable(self, events):
        if hasattr(events, "__iter__") and not isinstance(events, str):
            return self.nextEvent_callable_from_array(events)
        elif isinstance(events, function):
            return events

    def nextEvent_callable_from_array(self, events):
        """Generates callable that will find next event in events array"""
        events = np.append(np.atleast_1d(events), float("Inf"))
        def next_event_callable(time):
            if len(events) == 0:
                return float('Inf')
            else:
                i = np.argmax(events > time+1e-13)
                return events[i]
        return next_event_callable
    
class OutputGetterOMFMU2():
    def __init__(self, fmu, modelDescription, names_matches={}):
        self.fmu = fmu
        self.modelDescription = modelDescription
        self.names_matches = names_matches
        self.getters, self.all_names = self.find_all_matches(modelDescription, names_matches)
        self.df = pd.DataFrame(columns=self.all_names)

    def get_outputs(self, alias_name):
        d = self.getters[alias_name]
        type, names, refs, getter = d
        return np.array(getter(refs))
    
    def record_sample(self):
        for key, tp in self.getters.items():
            type, names, refs, getter = tp
            self.df.iloc[len(self.df.index)] = getter(refs)

    def find_all_matches(self, modelDescription, names_matches):
        getters = {}
        all_names = []
        new_tp = lambda t: (t, [], [], getattr(self.fmu, 'get' + t))
        for alias_name, pattern in names_matches.items():
            _pattern = re.compile(pattern)
            for variable in modelDescription.modelVariables:

                res = _pattern.search(variable.name)
                if res is not None:
                    _type = variable.type
                    _type = 'Integer' if _type == 'Enumeration' else _type
                    tp = getters.get(alias_name, new_tp(_type))
                    type, names, refs, getter = tp
                    assert type == _type, f"Inconsistent type matched in {alias_name}. Trigered by {variable.name} with patern {_pattern}"
                    names.append(variable.name)
                    all_names.append(variable.name)
                    refs.append(variable.valueReference)
                    getters[alias_name] = tp
            if len(getters.keys()) == 0:
                print(f'Warning: No outputs matched for alias {alias_name}')
        return getters, all_names

class OMFMU2model():
    "This runner works in kind of ModelExchange mode"
    def __init__(self, 
            fmu_path, rtol=None, start_time=0.0, stop_time=None, 
            maxStep=1.0, cvodemaxNumSteps=500, inputs=dict(),
            events=[], outputs_matches={}):
        self.fmu_path = fmu_path
        self.unzipdir = extract(fmu_path)
        self.model_description = read_model_description(self.unzipdir)
        self.fmu = FMU2Model(
            guid=self.model_description.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.model_description.coSimulation.modelIdentifier,
            instanceName=None)
        self.input_f_dict = inputs
        self.events = events
        
        if rtol is None:
            self.rtol = 1e-6
        self.start_time = start_time
        self.stop_time = stop_time
        self.maxStep = maxStep
        self.cvodemaxNumSteps = cvodemaxNumSteps
        self.output_interval = 0.1
        self.outputs = None
        self.outputs_matches = outputs_matches
        self.eps = 1e-13
        self.is_instantiated = False

    def instantiate(self, validate=True, start_values={}, visible=False, debug_logging=False, logger=None):
        callbacks=None #if logger == None
        self.fmu.instantiate(visible=visible, callbacks=callbacks, loggingOn=debug_logging)
        self.fmu.setupExperiment(startTime=self.start_time, stopTime=self.stop_time)
        self.input = InputSetterOMFMU2(self.fmu, self.model_description, self.input_f_dict, self.events)
        self.output = OutputGetterOMFMU2(self.fmu, self.model_description, self.outputs_matches)
        start_values = apply_start_values(self.fmu, self.model_description, start_values, settable=settable_in_instantiated)
        self.fmu.enterInitializationMode()
        start_values = apply_start_values(self.fmu, self.model_description, start_values, settable=settable_in_initialization_mode)
        self.time = self.start_time
        self.input.apply(self.time)
        self.fmu.exitInitializationMode() # FIXME: this causes SIGSEGV when testing

        self.newDiscreteStatesNeeded = True
        self.terminate_simulation = False
        while self.newDiscreteStatesNeeded and not self.terminate_simulation:
            (self.newDiscreteStatesNeeded,
             self.terminate_simulation,
             self.nominalsOfContinuousStatesChanged,
             self.valuesOfContinuousStatesChanged,
             self.nextEventTimeDefined,
             self.nextEventTime) = self.fmu.newDiscreteStates()
        
        if validate and len(start_values) > 0:
            raise Exception("The start values for the following variables could not be set: " +
                            ', '.join(start_values.keys()))
        
        # common solver constructor arguments
        self.solver_args = {
            'nx': self.model_description.numberOfContinuousStates,
            'nz': self.model_description.numberOfEventIndicators,
            'get_x': self.fmu.getContinuousStates,
            'set_x': self.fmu.setContinuousStates,
            'get_dx': self.fmu.getDerivatives,
            'get_z': self.fmu.getEventIndicators,
            'input': self.input
        }

        # alocate state array
        self.x = np.zeros(self.model_description.numberOfContinuousStates)
        self._px = self.x.ctypes.data_as(POINTER(c_double))

        #allocate state time-derivative array
        self.dx = np.zeros(self.model_description.numberOfContinuousStates)
        self._pdx = self.dx.ctypes.data_as(POINTER(c_double))

        self.solver = CVodeSolver(
            get_nominals=self.fmu.getNominalsOfContinuousStates,
            set_time=self.fmu.setTime,
            startTime=self.start_time,
            maxStep=self.maxStep,
            relativeTolerance=self.rtol,
            maxNumSteps=self.cvodemaxNumSteps,
            **self.solver_args)
        self.time = self.start_time

        self.recorder = Recorder(fmu=self.fmu,
            modelDescription=self.model_description,
            variableNames=self.outputs,
            interval=self.output_interval)
        self.recorder.sample(self.time)

        self.t_next = self.start_time
        self.n_fixed_steps = 0
        self.is_instantiated = True

    def destroy(self, ignore_warning=False):
        if self.is_instantiated:
            self.fmu.terminate()
            del self.solver
        elif not ignore_warning:
            print(f"Warning: Destroying non-instantiated FMU - {self.fmu_path}")
    
    def get_state(self):
        self.fmu.getContinuousStates(self._px, self.x.size)
        return self.x
    
    def set_state(self, x):
        px = x.ctypes.data_as(POINTER(c_double))
        self.fmu.setContinuousStates(px, x.size)

    def get_derivative(self):
        self.fmu.getDerivatives(self._pdx, self.dx.size)
        return self.dx
    
    def get_derivative_xt(self, x :np.ndarray, t: float):
        "Calculate time derivative from state x in time t"

        # insert internal state
        self.set_state(x)
        self.fmu.setTime(t)

        # calculate derivative at that state
        dx = self.get_derivative().copy()

        # return to original state
        self.set_state(self.x)
        self.fmu.setTime(self.time)

        return dx

    def do_step(self, output_interval=1.0, record_events=True, record=False):
        #prev_state = self.get_state().copy()
        #prev_time = self.time
        if self.time + self.eps >= self.t_next:  # t_next has been reached
            # integrate to the next grid point
            self.t_next = np.floor(self.time / output_interval) * output_interval + output_interval
            if self.t_next <= self.time + self.eps:
                self.t_next += output_interval

        # check if event happens between self.time and self.t_next
        # if so make the step only to the event point
        t_input_event = self.input.nextEvent(self.time)
        input_event = t_input_event <= self.t_next
        if input_event:
            self.t_next = t_input_event

        # check if time_event happens earlier that self.t_next
        # if so step to that time only
        time_event = self.nextEventTimeDefined and self.nextEventTime <= self.t_next
        if time_event:
            self.t_next = self.nextEventTime

        if self.t_next - self.time > self.eps:
            # do one step
            state_event, roots_found, self.time = self.solver.step(self.time, self.t_next)
        else:
            # skip
            state_event = False
            roots_found = []
            self.time = self.t_next

        # set the time in the fmu
        self.fmu.setTime(self.time)

        # apply continuous inputs
        self.input.apply(self.time, discrete=False)

        # check for step event, e.g. dynamic state selection
        if self.model_description.modelExchange.needsCompletedIntegratorStep:
            step_event, _ = self.fmu.completedIntegratorStep()
            step_event = step_event != fmi2False
        else:
            step_event = False

        # handle events
        if input_event or time_event or state_event or step_event:

            if record_events:
                # record the values before the event
                self.recorder.sample(self.time, force=True)

            self.fmu.enterEventMode()

            if input_event:
                self.input.apply(time=self.time, after_event=True)

            self.newDiscreteStatesNeeded = True

            # update discrete states
            while self.newDiscreteStatesNeeded and not self.terminate_simulation:
                (self.newDiscreteStatesNeeded,
                self.terminate_simulation,
                self.nominalsOfContinuousStatesChanged,
                self.valuesOfContinuousStatesChanged,
                self.nextEventTimeDefined,
                self.nextEventTime) = self.fmu.newDiscreteStates()

            # if self.terminate_simulation:
            #     break

            self.fmu.enterContinuousTimeMode()

            self.solver.reset(self.time)

            if record_events:
                # record values after the event
                self.recorder.sample(self.time, force=True)
            
        if self.time > self.recorder.lastSampleTime + self.eps:
            # record values for this step if not recorded by event 
            self.recorder.sample(self.time, force=True)

    def advance(self, output_interval=1.0, record_events=True):
        stop_time = self.time + output_interval

        # simulation loop
        while self.time + self.eps < stop_time:
            self.do_step(output_interval=output_interval, record_events=record_events)

    def simulate(self, stop_time=1.0, interval=0.01, record_events=True, verbose=False):
        n = int(np.floor(stop_time/interval))
        #TODO add ProgressBar here
        for i in range(n):
            self.advance(output_interval=interval, record_events=record_events)
            if verbose:
                print(f'Time: {self.time}')
        if self.time + self.eps < stop_time:
            interval = stop_time-self.time
            self.advance(output_interval=interval, record_events=record_events)
            if verbose:
                print(f'Time: {self.time}')

    def get_results(self):
        return self.recorder.result()

    def instantiate_and_simulate(self, **kwargs):
        self.destroy(ignore_warning=True)
        self.instantiate()
        self.simulate(**kwargs)
        res = self.get_results()
        self.destroy()
        return res