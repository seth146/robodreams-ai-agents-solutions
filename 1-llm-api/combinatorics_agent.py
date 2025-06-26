import os
import json
import math
from openai import OpenAI
from pprint import pprint
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Function Implementations
def calculate_combinations(n: int, m: int):
    """
    Calculate the number of combinations (n choose m).
    Formula: C(n,m) = n! / (m! * (n-m)!)
    """
    if m > n or m < 0 or n < 0:
        return {"error": "Invalid input: n and m must be non-negative integers with m <= n"}
    
    if m == 0 or m == n:
        return {"n": n, "m": m, "combinations": 1}
    
    # Calculate using math.comb for efficiency
    result = math.comb(n, m)
    return {"n": n, "m": m, "combinations": result}

def calculate_permutations(n: int, m: int):
    """
    Calculate the number of permutations P(n,m).
    Formula: P(n,m) = n! / (n-m)!
    """
    if m > n or m < 0 or n < 0:
        return {"error": "Invalid input: n and m must be non-negative integers with m <= n"}
    
    if m == 0:
        return {"n": n, "m": m, "permutations": 1}
    
    # Calculate using math.perm for efficiency
    result = math.perm(n, m)
    return {"n": n, "m": m, "permutations": result}

# Define custom tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_combinations",
            "description": "Calculate the number of combinations (n choose m). Use this when you need to find how many ways to choose m items from n items where order doesn't matter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Total number of items (n)",
                    },
                    "m": {
                        "type": "integer",
                        "description": "Number of items to choose (m)",
                    }
                },
                "required": ["n", "m"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_permutations",
            "description": "Calculate the number of permutations P(n,m). Use this when you need to find how many ways to arrange m items from n items where order matters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Total number of items (n)",
                    },
                    "m": {
                        "type": "integer",
                        "description": "Number of items to arrange (m)",
                    }
                },
                "required": ["n", "m"],
            },
        }
    },
]

available_functions = {
    "calculate_combinations": calculate_combinations,
    "calculate_permutations": calculate_permutations,
}


class CombinatoricsAgent:
    """A ReAct (Reason and Act) agent that handles combinatorics calculations."""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.max_iterations = 10  # Prevent infinite loops
        
    def run(self, messages: List[Dict[str, Any]]) -> str:
        """
        Run the ReAct loop until we get a final answer.
        
        The agent will:
        1. Call the LLM
        2. If tool calls are returned, execute them
        3. Add results to conversation and repeat
        4. Continue until LLM returns only text (no tool calls)
        """
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Call the LLM
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False
            )
            
            response_message = response.choices[0].message
            print(f"LLM Response: {response_message}")
            
            # Check if there are tool calls
            if response_message.tool_calls:
                # Add the assistant's message with tool calls to history
                messages.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in response_message.tool_calls
                    ]
                })
                
                # Process ALL tool calls
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    tool_id = tool_call.id
                    
                    print(f"Executing tool: {function_name}({function_args})")
                    
                    # Call the function
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**function_args)
                    
                    print(f"Tool result: {function_response}")
                    
                    # Add tool response to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": function_name,
                        "content": json.dumps(function_response),
                    })
                
                # Continue the loop to get the next response
                continue
                
            else:
                # No tool calls - we have our final answer
                final_content = response_message.content
                
                # Add the final assistant message to history
                messages.append({
                    "role": "assistant",
                    "content": final_content
                })
                
                print(f"\nFinal answer: {final_content}")
                return final_content
        
        # If we hit max iterations, return an error
        return "Error: Maximum iterations reached without getting a final answer."


def main():
    # Create a Combinatorics agent
    agent = CombinatoricsAgent()
    
    # Example 1: Simple combination calculation
    print("=== Example 1: Simple Combination ===")
    messages1 = [
        {"role": "system", "content": "You are a helpful AI assistant that can perform combinatorics calculations."},
        {"role": "user", "content": "How many ways can I choose 3 items from 10 items?"},
    ]
    
    result1 = agent.run(messages1.copy())
    print(f"\nResult: {result1}")
    
    # Example 2: Multiple calculations
    print("\n\n=== Example 2: Multiple Calculations ===")
    messages2 = [
        {"role": "system", "content": "You are a helpful AI assistant that can perform combinatorics calculations."},
        {"role": "user", "content": "Calculate both the combinations and permutations for choosing 2 items from 5 items. Explain the difference."},
    ]
    
    result2 = agent.run(messages2.copy())
    print(f"\nResult: {result2}")
    
    # Example 3: Real-world problem
    print("\n\n=== Example 3: Real-world Problem ===")
    messages3 = [
        {"role": "system", "content": "You are a helpful AI assistant that can perform combinatorics calculations."},
        {"role": "user", "content": "In a lottery where you need to pick 6 numbers from 49, how many different combinations are possible?"},
    ]
    
    result3 = agent.run(messages3.copy())
    print(f"\nResult: {result3}")
    
    # Example 4: Error handling
    print("\n\n=== Example 4: Error Handling ===")
    messages4 = [
        {"role": "system", "content": "You are a helpful AI assistant that can perform combinatorics calculations."},
        {"role": "user", "content": "What happens if I try to choose 5 items from 3 items?"},
    ]
    
    result4 = agent.run(messages4.copy())
    print(f"\nResult: {result4}")


if __name__ == "__main__":
    main() 