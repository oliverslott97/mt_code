from datetime import datetime
import os

__all__ = ['convert_date', 'test_make_dir']

def convert_date(value, output_type, input_format=None):
    
    if isinstance(value, output_type):
        if isinstance(value, datetime):
            return value
        elif isinstance(value, str) and input_format:
            try:
                parsed_date = datetime.strptime(value, input_format)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Invalid date format for input: {value}. Expected format: {input_format}")

    if isinstance(value, str) and output_type == datetime and input_format:
        try:
            parsed_date = datetime.strptime(value, input_format)
            return parsed_date
        except ValueError:
            raise ValueError(f"Invalid date format for input: {value}. Expected format: {input_format}")

    if isinstance(value, datetime) and output_type == str:
        return value.strftime('%Y-%m-%d')

    raise ValueError(f"Invalid input or output type. Got {type(value)} as input and {output_type} as output type.")

def test_make_dir(path):

    if os.path.isfile(path):
        raise ValueError(f"The path {path} is an already existing file, not a directory.")
    if not os.path.exists(path):
        os.makedirs(path)
    
    return path
