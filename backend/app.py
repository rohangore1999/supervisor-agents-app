from flask import Flask, request, jsonify
import re
import json
from typing import Literal # Typing utilities for type hints and better code readability
from pydantic import BaseModel, Field  # `BaseModel` is the base class used to create data models, `Field` is used to provide additional metadata
from langchain_core.messages import HumanMessage # Human message classes for handling human messages in LangChain
from langchain_community.tools.riza.command import ExecPython # Riza's ExecPython for executing Python code dynamically
from langchain_groq import ChatGroq # Interface for using the ChatGroq platform for advanced language model capabilities
from langgraph.types import Command # LangGraph types for extending commands and functionalities
from langgraph.graph import StateGraph, START, END, MessagesState # Graph-related utilities for building workflows and state machines
from langgraph.prebuilt import create_react_agent # Prebuilt tools and agents for streamlined development
from pprint import pprint # Utilities for debugging and displaying complex data structures in an organized way
from langchain_community.tools import DuckDuckGoSearchRun ## Research tool

app = Flask(__name__)

# Initialize LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

# Initialize Tools
duck_duck_go = DuckDuckGoSearchRun()
tool_code_interpreter = ExecPython()
tools = [duck_duck_go, tool_code_interpreter]

# Define a Supervisor class to specify the next worker in the pipeline
"""
BaseModel in args: 
- This inheritance provides automatic data validation capabilities
- The Validator class becomes a schema definition for structured data
"""
class Supervisor(BaseModel):
    next: Literal["enhancer", "researcher", "coder"] = Field(
        description="Specifies the next worker in the pipeline: "
                    "'enhancer' for enhancing the user prompt if it is unclear or vague, "
                    "'researcher' for additional information gathering, "
                    "'coder' for solving technical or code-related problems."
    )
    reason: str = Field(
        description="The reason for the decision, providing context on why a particular worker was chosen."
    )

# Define the supervisor node
def supervisor_node(state: MessagesState) -> Command[Literal["enhancer", "researcher", "coder"]]:
    """
    Supervisor node for routing tasks based on the current state and LLM response.
    Args:
        state (MessagesState): The current state containing message history.
    Returns:
        Command: A command indicating the next state or action.
    """
    
    system_prompt = ('''You are a workflow supervisor managing a team of three agents: Prompt Enhancer, Researcher, and Coder. Your role is to direct the flow of tasks by selecting the next agent based on the current stage of the workflow. For each task, provide a clear rationale for your choice, ensuring that the workflow progresses logically, efficiently, and toward a timely completion.

    **Team Members**:
    1. Enhancer: Use prompt enhancer as the first preference, to Focuse on clarifying vague or incomplete user queries, improving their quality, and ensuring they are well-defined before further processing.
    2. Researcher: Specializes in gathering information.
    3. Coder: Handles technical tasks related to caluclation, coding, data analysis, and problem-solving, ensuring the correct implementation of solutions.

    **Responsibilities**:
    1. Carefully review each user request and evaluate agent responses for relevance and completeness.
    2. Continuously route tasks to the next best-suited agent if needed.
    3. Ensure the workflow progresses efficiently, without terminating until the task is fully resolved.

    Your goal is to maximize accuracy and effectiveness by leveraging each agent’s unique expertise while ensuring smooth workflow execution.
    ''')
    
    # Prepare messages by appending the system prompt to the message history
    messages = [
        {"role": "system", "content": system_prompt},  # System-level instructions or context
    ] + state["messages"]  # Append previous messages from the state

    # Invoke the language model with structured output.
    response = llm.with_structured_output(Supervisor).invoke(messages)
    
    # Extract the 'next' routing decision and the 'reason' from the response
    goto = response.next
    reason = response.reason
    
    # Debug logging to trace responses and transitions
    print(f"Current Node:  Supervisor -> Goto: {goto}")
    
    # Updating the state with the supervisor's response and routing to the next node.
    return Command(
        update={
            "messages": [
                # Append the reason (supervisor's response) to the state, tagged with "supervisor"
                HumanMessage(content=reason, name="supervisor")
            ]
        },
        goto=goto,  # Specify the next node in the workflow
    )

# Define the enhancer node
def enhancer_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    Enhancer node for refining and clarifying user inputs.

    Args:
        state (MessagesState): The current state containing the conversation history.

    Returns:
        Command: A command to update the state with the enhanced query and route back to the supervisor.
    """
    # Define the system prompt to guide the LLM in query enhancement
    system_prompt = (
        "You are an advanced query enhancer. Your task is to:\n"
        "Don't ask anything to the user, select the most appropriate prompt"
        "1. Clarify and refine user inputs.\n"
        "2. Identify any ambiguities in the query.\n"
        "3. Generate a more precise and actionable version of the original request.\n"
    )

    # Combine the system prompt with the current conversation messages
    messages = [
        {"role": "system", "content": system_prompt},  # Provide context for the LLM
    ] + state["messages"]  # Include the conversation history for context

    # Use the LLM to process the messages and generate an enhanced query
    enhanced_query = llm.invoke(messages)
    print(f"Current Node: Prompt Enhancer -> Goto: Supervisor")
    # print(f"Response: {enhanced_query}")
    # Return a command to update the state with the enhanced query and route back to the supervisor
    return Command(
        update={
            "messages": [  # Append the enhanced query to the message history
                HumanMessage(
                    content=enhanced_query.content,  # Content of the enhanced query
                    name="enhancer"  # Name of the node processing this message
                )
            ]
        },
        goto="supervisor",  # Route to the supervisor for further processing
    )

# Define the research node
def research_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Research node for leveraging a ReAct agent to process research-related tasks.

    Args:
        state (MessagesState): The current state containing the conversation history.

    Returns:
        Command: A command to update the state with the research results and route to the validator.
    """
    # Create a ReAct agent specialized for research tasks
    research_agent = create_react_agent(
        llm,  # The language model instance used by the agent
        tools=[duck_duck_go],  # List of tools the agent can utilize
        state_modifier="You are a researcher. Focus on gathering information and generating content. Do not perform any other tasks"  # Instruction to restrict the agent's behavior
    )
    
    # Invoke the research agent with the current state to perform its task
    result = research_agent.invoke(state)
    
    print(f"Current Node: Researcher -> Goto: Validator")
    
    # Extract the last message from the result and update the state
    return Command(
        update={
            "messages": [  # Append the research results to the message history
                HumanMessage(
                    content=result["messages"][-1].content,  # Content of the agent's response
                    name="researcher"  # Name of the node generating this message
                )
            ]
        },
        goto="validator",  # Route back to the supervisor for further processing
    )

# Define the code node
def code_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Coder node for leveraging a ReAct agent to process analyzing, solving math questions, and executing code.

    Args:
        state (MessagesState): The current state containing the conversation history.

    Returns:
        Command: A command to update the state with the research results and route to the validator.
    """
    # Create a specialized ReAct agent for coding and problem-solving tasks
    code_agent = create_react_agent(
        llm,
        tools=[tool_code_interpreter],
        state_modifier=(
            "You are a coder and analyst. Focus on mathematical caluclations, analyzing, solving math questions, "
            "and executing code. Handle technical problem-solving and data tasks."
        )
    )

    # Invoke the agent with the current state to process the input and perform its task
    result = code_agent.invoke(state)

    # Debug logging to trace responses and node transitions
    print(f"Current Node: Coder -> Goto: validator")
    # print(f"Response:", result)

    # Return a command to update the state and move to the 'validator' node
    return Command(
        update={
            "messages": [
                # Append the last message (agent's response) to the state, tagged with "coder"
                HumanMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        # Specify the next node in the workflow: "validator"
        goto="validator",
    )

# Define a Validator class for structured output from the LLM
"""
BaseModel in args: 
- This inheritance provides automatic data validation capabilities
- The Validator class becomes a schema definition for structured data
"""
class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Specifies the next worker in the pipeline: 'supervisor' to continue or 'FINISH' to terminate."
    )
    reason: str = Field(
        description="The reason for the decision."
    )

# Define the validator node function to check the user question and the answer
def validator_node(state: MessagesState) -> Command[Literal["supervisor", "__end__"]]:
    """
    Validator node for checking if the question and the answer are appropriate.

    Args:
        state (MessagesState): The current state containing message history.

    Returns:
        Command: A command indicating whether to route back to the supervisor or end the workflow.
    """
    
    # System prompt providing clear instructions to the validator agent
    system_prompt = '''
        You are a workflow validator. Your task is to ensure the quality of the workflow. Specifically, you must:
        - Review the user's question (the first message in the workflow).
        - Review the answer (the last message in the workflow).
        - If the answer satisfactorily addresses the question, signal to end the workflow.
        - If the answer is inappropriate or incomplete, signal to route back to the supervisor for re-evaluation or further refinement.
        Ensure that the question and answer match logically and the workflow can be concluded or continued based on this evaluation.

        Routing Guidelines:
        1. 'supervisor' Agent: For unclear or vague state messages.
        2. Respond with 'FINISH' to end the workflow.
    '''
    # Extract the first (user's question) and the last (agent's response) messages
    user_question = state["messages"][0].content
    agent_answer = state["messages"][-1].content

    # Prepare the message history with the system prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": agent_answer},
    ]

    # Invoke the LLM with structured output using the Validator schema
    response = llm.with_structured_output(Validator).invoke(messages)

    # Extract the 'next' routing decision and the 'reason' from the response
    goto = response.next
    reason = response.reason

    # Determine the next node in the workflow
    if goto == "FINISH" or goto == END:
        goto = END  # Transition to the termination state
        print("Transitioning to END")  # Debug log to indicate process completion
    else:
        print(f"Current Node: Validator -> Goto: Supervisor")  # Log for routing back to supervisor
    # Debug logging to trace responses and transitions
    # print(f"Response: {response}")
    # Return a command with the updated state and the determined routing destination
    return Command(
        update={
            "messages": [
                # Append the reason (validator's response) to the state, tagged with "validator"
                HumanMessage(content=reason, name="validator")
            ]
        },
        goto=goto,  # Specify the next node in the workflow
    )

# Initialize the state graph
builder = StateGraph(MessagesState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("enhancer", enhancer_node)
builder.add_node("researcher", research_node)
builder.add_node("coder", code_node)
builder.add_node("validator", validator_node)

builder.add_edge(START, "supervisor")

graph = builder.compile()

# Route to render HTML interface
@app.route('/')
def index():
    return "App is running"

# API endpoint to chat user input
@app.route('/api/chat', methods=['POST'])
def process_input():
    try:
        data = request.json
        user_input = data.get('input')
        
        if not user_input:
            return jsonify({"error": "No input provided"}), 400
        
        inputs = {
            "messages": [
                HumanMessage(content=user_input)
            ]
        }
        
        outputs = []
        
        for output in graph.stream(inputs):
            """
            Instead of waiting for all processing to finish before returning anything (like waiting for a complete meal), the system:
                - Returns each piece of output as soon as it's ready
                - In this code, each time a node/agent (coder, researcher, enhancer) completes its task, it immediately provides its result
                - The loop processes each result as it arrives, rather than waiting for all components to finish
            """
            node_outputs = {}
            for key, value in output.items():
                """
                dictionary where keys represent node names and values contain their respective outputs
                """
                if value is not None:
                    node_outputs[key] = str(value)
            outputs.append(node_outputs)
        
        # Extract just the answer from the response
        answer = "No answer found"
        
        # First try to get the answer from the coder's response
        for output in outputs:
            if "coder" in output:
                coder_value = output["coder"]
                content_match = re.search(r"content='([^']*)'", coder_value)
                if content_match and content_match.group(1).strip():
                    answer = content_match.group(1)
                    break
        
        # If coder didn't provide an answer, check researcher
        if answer == "No answer found" or answer == "":
            for output in outputs:
                if "researcher" in output:
                    researcher_value = output["researcher"]
                    content_match = re.search(r"content='([^']*)'", researcher_value)
                    if content_match and content_match.group(1).strip():
                        answer = content_match.group(1)
                        break
        
        # If still no answer, check enhancer
        if answer == "No answer found" or answer == "":
            for output in outputs:
                if "enhancer" in output:
                    enhancer_value = output["enhancer"]
                    content_match = re.search(r"content='([^']*)'", enhancer_value)
                    if content_match and content_match.group(1).strip():
                        answer = content_match.group(1)
                        break
        
        return jsonify({
            "answer": answer,
            "all_outputs": outputs
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)  