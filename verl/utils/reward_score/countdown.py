import re
import random
import ast
import operator


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
        if '=' in final_answer:
            final_answer = final_answer.split('=', 1)[0].strip()
        else:
            final_answer = final_answer
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # another possible form: 1 + 2 -3 = 0.
        equation_str = equation_str.split('=', 1)[0].strip()

        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        print(f"Available numbers: {available_numbers}")
        print(f"Numbers in equation: {numbers_in_eq}")
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.0, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 16) == 1

    # only print correct solution for checking.
    if do_print:
        print(f"--------------------------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        ret_score = 0.0
        return {
            "score": ret_score,
            "acc": ret_score
        }
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation: {equation}......")
        ret_score = format_score
        return {
            "score": ret_score,
            "acc": ret_score
        }
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            ret_score = format_score
            return {
                "score": ret_score,
                "acc": ret_score
            }
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
                print(f"Target: {target} | Numbers: {numbers}")
                print(f"Extracted equation: {equation}")
                print(f"Solution string: {solution_str}")
            ret_score = score
            return {
                "score": ret_score,
                "acc": ret_score
            }
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            ret_score = format_score
            return {
                "score": ret_score,
                "acc": ret_score
            }
    except:
        if do_print:
            print(f"Error evaluating equation")
        ret_score = format_score 
        return {
            "score": ret_score,
            "acc": ret_score
        }

    return {
        "score": ret_score,
        "acc": ret_score
    }