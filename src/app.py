from langgraph.graph import StateGraph, END
from graph_state import GraphState
from nodes import retrieve_schema, generate_sql, validate_and_execute

# Define the workflow graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve_schema", retrieve_schema)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("validate_and_execute", validate_and_execute)

# Define the entry point and edges
workflow.set_entry_point("retrieve_schema")
workflow.add_edge("retrieve_schema", "generate_sql")
workflow.add_edge("generate_sql", "validate_and_execute")

# Define the conditional edge for the last step
def should_continue(state: GraphState):
    """Conditional logic to decide if the workflow should stop."""
    if state.get("error_message"):
        # You could add a retry loop here for a more advanced agent
        return "end" 
    else:
        return "end"

workflow.add_conditional_edges(
    "validate_and_execute",
    should_continue,
    {"end": END}
)

# Compile the graph
app = workflow.compile()

# --- Main function to run the workflow ---
if __name__ == "__main__":
    initial_state = {
        "user_question": "List all the product verticals",
        "sql_query": "",
        "query_result": [],
        "error_message": ""
    }
    
    # Run the graph
    for state in app.stream(initial_state):
        print(state)