import pandas as pd

def apply_transform_step(df, step):
        """Apply a single transform step to a dataframe"""
        step_type = step.get('type', '')
        operation = step.get('operation', '')
        
        try:
            if step_type == 'python_transform':
                func = step.get('function')
                if func is None:
                    print("No 'function' provided in custom_function step")
                elif isinstance(func, str):
                    func = eval(func)
                elif callable(func):
                    pass
                else:
                    print(f"Invalid function argument provided in python_transform step: {func}")
                    return df
                return func(df.copy())

            elif step_type == 'column_selection' or operation == 'select':
                column_pattern = step.get('column_pattern', '.*')
                import re
                matching_cols = [col for col in df.columns if re.match(column_pattern, col)]
                # Always preserve t_timestamp column for time-based operations
                if 't_timestamp' in df.columns and 't_timestamp' not in matching_cols:
                    matching_cols.append('t_timestamp')
                if matching_cols:
                    return df[matching_cols]
                # If no matches but t_timestamp exists, return at least t_timestamp
                if 't_timestamp' in df.columns:
                    return df[['t_timestamp']]
                return df
            
            elif step_type == 'time_aggregation' or operation == 'resample':
                frequency = step.get('frequency', '1D')
                method = step.get('method', 'mean')
                print(f"Resampling with frequency: {frequency} and method: {method}")
                if method == 'mean':
                    result = df.resample(frequency).mean()
                elif method == 'last':
                    result = df.resample(frequency).last()
                elif method == 'first':
                    result = df.resample(frequency).first()
                elif method == 'sum':
                    result = df.resample(frequency).sum()
                else:
                    result = df.resample(frequency).mean()
                return result.reset_index()
            
            elif step_type == 'column_transform' or operation in ['add_constant', 'multiply_constant', 'diff']:
                column_pattern = step.get('column_pattern', '.*')
                import re
                matching_cols = [col for col in df.columns if re.match(column_pattern, col)]
                # Exclude t_timestamp from transformations (it should not be modified)
                matching_cols = [col for col in matching_cols if col != 't_timestamp']
                
                if operation == 'add_constant':
                    constant = step.get('constant', 0)
                    suffix = step.get('new_column_suffix', '_transformed')
                    for col in matching_cols:
                        df[col + suffix] = df[col] + constant
                
                elif operation == 'multiply_constant':
                    constant = step.get('constant', 1)
                    suffix = step.get('new_column_suffix', '_transformed')
                    for col in matching_cols:
                        df[col + suffix] = df[col] * constant
                
                elif operation == 'diff':
                    suffix = step.get('new_column_suffix', '_diff')
                    for col in matching_cols:
                        df[col + suffix] = df[col].diff()
                        df.iloc[0, df.columns.get_loc(col + suffix)] = df.iloc[0, df.columns.get_loc(col)]
                
                # Ensure t_timestamp is preserved
                if 't_timestamp' not in df.columns and 't_timestamp' in df.index.names:
                    df = df.reset_index()
                
                return df
            
            elif step_type == 'row_selection' or operation == 'row_select':
                method = step.get('method', 'nearest')
                if method == 'earliest':
                    return df.iloc[[0]]
                elif method == 'latest':
                    return df.iloc[[-1]]
                elif method == 'nearest':
                    target = step.get('target_time')
                    if target is None:
                        print("No 'target_time' provided for nearest row selection")
                        return df
                    target_ts = pd.Timestamp(target, utc=True)
                    idx = df.index.get_indexer([target_ts], method='nearest')[0]
                    return df.iloc[[idx]]
                else:
                    print(f"Unknown row_selection method: {method}")
                    return df

            elif step_type == 'to_pie' or operation == 'to_pie':
                #assert len(df.index) == 1, "to_pie transform requires a single row dataframe"
                labels = df.columns.tolist() 
                values = df.values.tolist()[0]
                print(len(labels), len(values))
                return pd.DataFrame(data={
                    'labels': labels,
                    'values': values
                })

            elif step_type == 'formula':
                formula = step.get('formula', '')
                new_column = step.get('new_column_name', 'result')
                try:
                    df = df.copy()
                    df[new_column] = df.eval(formula)
                    return df
                except:
                    return df
            
            return df
        except Exception as e:
            print(f"Error applying step: {e}")
            return None

def _parse_timestamp_column(df):
    """Hidden first step: Parse t_timestamp column using the same logic as figure_updaters"""
    if 't_timestamp' not in df.columns:
        print("t_timestamp column not found in dataframe")
        return df

    df = df.copy()
    # Apply parsing to t_timestamp column
    df['t_timestamp'] = pd.to_datetime(df['t_timestamp'], unit='s', origin='unix', utc=True)
    df = df.set_index('t_timestamp')
    
    return df

def apply_transform(df, transform, n_steps=None):
    df = _parse_timestamp_column(df)
    for i, step in enumerate(transform['steps']):
        df = apply_transform_step(df, step)
        if df is None:
            return None
        if n_steps is not None and i >= n_steps:
            break
    return df