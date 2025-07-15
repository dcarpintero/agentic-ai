# The Agentic Shift - Design Patterns for Building AI Systems

*Originally published on 5 July 2025 at https://tech.diegocarpintero.com/blog/agentic-shift-design-patterns-for-building-ai-systems/*

## Introduction

Driven by advances in frontier models such as (i) enhanced contextual understanding, (ii) integration of multimodal capabilities, and (iii) inference-time scaling [1], AI-based applications are evolving from reactive, static assistants that respond to explicit inputs toward autonomous systems that augment our knowledge and capabilities.

Central to this transformation are AI Agents, systems that use a language model to interpret natural language instructions, engage with their environment, and take actions toward achieving a goal. Agents can operate with varying degrees of autonomy and collaborate with other agents and humans. Unlike traditional software that follows predetermined logic, agents adapt their behavior based on observations, past interactions, and reflection. This flexibility allows agents to proactively respond to changing conditions and coordinate across multiple systems. However, this approach introduces challenges around design, verification, scalability in production, cost management, and maintenance.

This article provides a comprehensive guide to building and orchestrating AI Agents that reason, plan, and act using six foundational design patterns: Evaluator-Optimizer, Context-Augmentation, Prompt-Chaining, Parallelization, Routing, and Orchestrator-Workers.

## Table of Contents

- [Evaluator-Optimizer Pattern](#evaluator-optimizer-pattern)
- [Context-Augmentation Pattern](#context-augmentation-pattern)
- [Prompt-Chaining Workflow](#prompt-chaining-workflow)
- [Parallelization Workflow](#parallelization-workflow)
- [Routing Workflow](#routing-workflow)
- [Orchestrator-Workers Workflow](#orchestrator-workers-workflow)
- [References](#references)

## Evaluator-Optimizer Pattern

This pattern draws inspiration from *self-reflection*, a fundamental process of human problem-solving that involves introspective examination of thoughts, actions, and experiences to gain insight and improve results. As an analogy, consider how writers prepare an essay: drafting initial ideas, reviewing their work to identify unclear passages or verify facts, and then revising it accordingly. This cycle continues until the author is satisfied with the result. The *Evaluator-Optimizer* pattern [2] [3] [4] applies a similar iterative refinement to LLM response generation.

In practice, it introduces a cognitive layer where LLMs enhance their responses through systematic feedback and refinement. Rather than accepting first-attempt outputs, it establishes a structured process for AI systems to self-reflect on past actions and implement targeted improvements.

### Architecture

Its name reflects its core architecture: the *Evaluator* provides a critical assessment of the output, while the *Generator / Optimizer* transforms such actionable feedback and guidance into concrete improvements. The pattern implements the following workflow:

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/J6jEhe9vnSwo-4X9RDKYy.png">
  <figcaption style="text-align: center;">Evaluator-Optimizer Pattern</figcaption>
</figure>

- *Generation*: The LLM generates an initial response, or completes a task using its standard capabilities. This first attempt serves as the baseline output, though it may contain errors, lack clarity, or miss relevant facts.
- *Reflection - Evaluator*: The same LLM (or a specialized evaluator model) examines the generated output and systematically evaluates it against given requirements, guidelines, or external observations.
- *Refinement - Optimizer*: Translates reflective feedback into concrete improvements. This might involve restructuring content, adding missing elements, or correcting identified issues.
- *Iteration*: The process repeats until the output meets predetermined quality criteria or stopping conditions. Each iteration can reveal new improvement opportunities.

These steps can be carried out by a single LLM at each stage, or distributed across specialized LLMs optimized for evaluation versus generation tasks.

### Use Cases

The intended task typically requires to generate a high-quality output characterized by completeness, correctness, and accuracy:

- *Critical Software Code Generation*: Generating code, validating it against test cases, debugging errors, and refining for style and efficiency. 
- *Writing*: Improving linguistic flow, reducing bias, and ensuring alignment with a targeted audience.
- *Analytical Work*: Complex tasks requiring nuanced judgment and multiple perspectives.
- *Trustworthiness*: Cross-referencing generated content against search engines or trusted knowledge bases to ensure accuracy.

### Considerations

- *Balancing Cost-Benefit*: The most critical implementation decision involves determining when quality gains justify additional computational costs. Each iteration increases both latency and token usage, making it essential to define clear criteria for when this approach adds value.
- *Effective Stopping Criteria*: Poorly designed stopping conditions can trap systems in suboptimal states - either terminating too early and missing obvious improvements, or making marginal refinements indefinitely. Robust implementations incorporate multiple safeguards:
  - *Quality gates (such as code passing test suites)*
  - *Maximum iteration boundaries*
  - *Token budget constraints*
  - *Similarity checks to identify when successive versions show minimal, meaningful changes*
- *Actionable Feedback*: The feedback provided provided by the evaluator must be specific and concrete to guide the Generator / Optimizer.

## Context-Augmentation Pattern

This pattern addresses a fundamental limitation of LLMs: their capabilities remain static, limited by training data. Consider how we tackle complex problems in real-life: rather than relying solely on memory, we consult databases, check online information, and use specialized tools.

The *Context Augmentation Pattern* brings this approach to LLMs by dynamically expanding their context beyond pre-trained knowledge. Building upon *function-calling patterns* [5] [6] [7], it enables LLMs to connect and work with tools and external systems. These interactions have been standardized by the Model Context Protocol [8], making the pattern reliable for production applications. Its primary advantages are flexibility and extensibility, as new capabilities can be easily added to a model.

### Architecture

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/hLYAAg9lUccYE968DNVKP.png">
  <figcaption style="text-align: center;">Context-Augmentation Pattern</figcaption>
</figure>

- *Requirements Identification*: The LLM analyzes the intended task and identifies needs that exceed its internal capabilities.
- *Resource Selection*: From available resources, the model selects appropriate tools, systems or data sources (based on the specific need). This might involve choosing between a calculator for mathematical operations, a web search, or a database query.
- *Context Augmentation*: The LLM interacts with tools and external systems to integrate additional information into its working context. This step fundamentally expands the model's knowledge for the current task.
- *Synthesis*: The augmented context enables the model to provide responses that combine its reasoning capabilities with task-specific information.
- *Orchestration*: Complex tasks may require multiple interactions with resources, wherein each result informs subsequent resource selection and usage.

### Use Cases

- *Real-Time Information*: Accessing data that changes frequently and was not available during training such as weather conditions, stock market data, and recent news articles.
- *Computational Tasks*: Performing complex mathematical operations or data analysis requiring precision beyond what language models can provide.
- *Knowledge Integration*: Searching company documentation, accessing customer support databases, or querying technical specifications.
- *Device Control*: Managing IoT ecosystems such as smart home devices and industrial sensors.
- *Enterprise Orchestration*: Integrating specialized services for business tasks such as payment processing, communication management, and software development workflows.

### Considerations

- *Security and Access Control*: Tool integration and interactions with MCP Servers introduce significant security risks. Robust implementations require careful authentication, authorization, and sandboxing to prevent unintended access to sensitive systems or data.
- *Reliability and Error Handling*: External dependencies can fail or return unexpected results. This requires fallback strategies and error recovery mechanisms.
- *Operational Challenges*: Resource selection strategies become critical with multiple available resources — poor choices might lead to inefficient workflows and suboptimal results. Additionally, API calls and third-party services can introduce additional costs.
- *Technical Constraints*: Context window limitations necessitate balancing information retrieval with context management, potentially requiring summarization or selective retention.

## Prompt-Chaining Workflow

The Prompt-Chaining-Workflow [9] evolved from Chain-Of-Thought prompting (CoT) by applying *cognitive decomposition*, a common approach in problem-solving where complex tasks are broken down into smaller, manageable subtasks. In practice, an AI Engineer generates a sequence of specialized prompts for each subtask, where the output of one prompt becomes the input for the next, thus aggregating the gains from each step. Rather than overloading the model with a single complex instruction, it creates a structured pipeline that leverages the model's full attention at each step while maintaining context continuity throughout the process.

Beyond achieving better performance, prompt chaining enhances the transparency, controllability, and flexibility of LLM applications. Moreover, this structured approach enables easier debugging of model responses, targeted performance improvements at specific stages within the workflow, and support for different paths.

### Architecture

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/aXrDCyM6SlHvNN8hA5GOb.png">
  <figcaption style="text-align: center;">Prompt-Chaining Workflow</figcaption>
</figure>

- *Task Decomposition*: A complex task is analyzed and broken down into logical subtasks. This decomposition can be done manually or with LLM assistance using meta-prompting techniques.
- *Sequential Processing*: Each subtask is handled by a specialized prompt within a dedicated LLM call, allowing focused attention on the specific subtask.
- *Information Handoff*: Outputs from each prompt are structured to provide the necessary context for the next step. This may involve data transformation, formatting, or filtering to ensure clean transitions between stages.
- *Context Preservation*: The chain maintains relevant context across steps while avoiding information overload. Each step should receive only the information necessary for its specific task, preventing context dilution.
- *Gates*: Conditional checkpoints can be integrated between steps to validate outputs, route workflows based on content analysis, or terminate the chain when certain conditions are met.

### Use Cases

- *Documents Analysis*: Breaking down large documents into manageable tasks such as keyword extraction, summarization, and classification.
- *Content Creation*: Including tasks such as drafting, editing, formatting, and translation.
- *Research Assistance*: For systematic evaluation of hypotheses, results analysis, and literature review.

### Considerations

- *Balancing Cost-Benefit*: Prompt chaining can produce more favorable outcomes compared to single-prompt approaches. However, this improvement comes with increased computational costs and latency.
- *Selective Context Management*: Outputs from previous steps must contain only the information necessary for the next task, as verbosity or irrelevant information can exceed context window limits and disrupt subsequent steps.
- *Intermediate Validations*: Mistakes can cascade and compound through the entire chain. To mitigate these errors, it is recommended to implement quality gates before passing to the next step, this might include:
  - *Programmatic Checks (e.g., content length, format validation)*
  - *LLM-based output validation (e.g., for factual accuracy)*
  - *Confidence scoring*
- *Error Handling*: Failing outputs can be handled with:
  - *Retry*
  - *Re-prompting including the error details*
  - *Fallback Mechanisms such as logging and termination, or routing*.

## Parallelization Workflow

This workflow follows the *divide and conquer* approach and distributes the workload among agents to process independent tasks concurrently. This mirrors the scatter-gather pattern in distributed computing, where a task is split and assigned to multiple workers, and their individual results are then aggregated to form a complete output. In agentic AI systems, this workflow is typically implemented in two key variations [10]:

- Sectioning: Breaking a task into independent subtasks that run in parallel.
- Voting: Running the same task multiple times to obtain diverse outputs.

Parallelization is typically used with predictable, divisible workloads to improve accuracy and speed up completion.  

### Architecture

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/eAb_9YxGYZKS3UIZk5UPd.png">
  <figcaption style="text-align: center;">Parallelization Workflow</figcaption>
</figure>

- *Task Decomposition*: Analyzes the incoming task and identifies independent subtasks that can be processed simultaneously.
- *Parallel Processing:* Subtasks are allocated and processed by workers, each receiving specific instructions, context, and success criteria.
- *Aggregation:*  Results from all workers are aggregated through various synthesis methods to produce a coherent final output.

### Use Cases

#### Sectioning

- *Content Guardrails*: One model instance processes user queries while another screens them for inappropriate content.
- *Automated Evals*: Multiple LLM calls evaluate different aspects of model performance on given prompts.
- *Document Analysis*: Assigning document sections like chapters to different workers, then synthesizing or aggregating the outputs.

#### Voting

- *Code Security Review*: Different prompts review and flag code for vulnerabilities, with multiple perspectives increasing detection accuracy.
- *Content Moderation*: Multiple prompts evaluate different aspects of the input or require different vote thresholds to balance false positives and negatives.

### Considerations

- *Task Independence*: Ensuring subtasks are truly independent. Tasks with sequential dependencies are better suited for prompt chaining patterns.
- *Decomposition Strategy*:
  - *Sectioning/Sharding*: Splitting large inputs like a long document or a dataset into smaller chunks, with each chunk being processed in parallel
  - *Aspect-Based*: For tasks requiring analysis of multiple independent facets
  - *Identical Tasks for Diversity or Voting*: Running the same core task multiple times with varied prompts or models with the goal of achieving a more reliable result
- *Aggregation Strategy*: After the parallel tasks are completed, outputs can be combined using methods like:
  - *Concatenation*: Joining the outputs together, like appending individual chapter summaries
  - *Comparison and Selection*: Choosing among alternative outputs for the same task
  - *Voting/Majority Rule*: Consensus-based decision making. When multiple agents independently perform the same task, the most frequently occurring output is chosen as the final answer, boosting accuracy.
  - *Synthesizer*: A dedicated LLM creates a coherent final response from diverse outputs

## Routing Workflow

Routing evaluates an input and delegates it to a specialized function, tool, or sub-agent. This workflow enables: separation of concerns, resource optimization, flexibility, and scalability.

### Architecture

It operates through a two-step architecture: task classification and task dispatch.

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/vSfh8a3U7ToxcIJawO_97.png">
  <figcaption style="text-align: center;">Routing Workflow</figcaption>
</figure>

- *Task Classification*: The system analyzes incoming requests to determine their type, category, intent, or complexity level. It can be implemented through various methods:
  - *Rule-based*: Using predefined logic, keyword matching, or conditional statements
  - *LLM-based classification*: Leveraging contextual understanding for more nuanced categorization
  - *Hybrid methods*: Combining both approaches

- *Task Dispatch*: Based on the classification results, the system routes the input. In practice, the processing path selection depends on:
  - *Agent specialization and expertise*
  - *Resource requirements*
  - *Performance characteristics (speed, cost, accuracy)*
  - *Current system load and capacity*

### Use Cases

- *E-Commerce / Customer Service*: Automatically directing different types of user requests into different downstream processes, prompts, and tools.
- *Workload Branching*: Distributing inputs based on type and complexity, routing straightforward queries to lightweight models and complex questions to more capable models to optimize cost and speed.
- *Multi-Domain Q&A*: Directing domain-specific questions to specialized knowledge agents.

### Considerations

- *Classification Accuracy*: The workflow's effectiveness depends entirely on accurate input classification. If a task is misclassified, it might be sent to the wrong agent, leading to inefficiency or incorrect results.
- *Specialization Balance*: While specialization improves performance for specific tasks, the system should maintain appropriate generalization and flexibility to ensure that edge cases and novel inputs find their way to the most appropriate processing path.
- *Dynamic Routing*: Advanced implementations incorporate real-time system state into routing decisions, considering factors such as agent availability, current workload, and performance metrics.
- *Scalability and Maintenance*: As systems grow, the number of potential routing paths might increase significantly.

## Orchestrator-Workers Workflow

In this workflow, a central LLM dynamically decomposes tasks, delegates them to worker agents, and synthesizes their outputs into coherent results.

### Architecture

The *Orchestrator* is the lead agent providing intelligent coordination and synthesis, while the *Workers* execute specialized subtasks.

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/kC70rEEGNOxEODMYuJvPF.png">
  <figcaption style="text-align: center;">Orchestrator-Workers Workflow</figcaption>
</figure>

- *Task Analysis*: The *Orchestrator* analyzes incoming requests in real-time, dynamically determining necessary subtasks rather than following pre-defined workflows. This  considers task complexity, available resources, and optimal decomposition.
- *Delegation*: The *Orchestrator* selects appropriate workers to delegate subtasks, providing each worker with clear objectives, output formats, and tool guidance.
- *Result Synthesis*: The *Orchestrator* collects outputs from all workers and synthesizes them into a unified, coherent response.

### Use Cases

This workflow is well-suited for scenarios where subtasks are not pre-defined, but determined dynamically by the *Orchestrator* based on the input, such as:

- *Business Intelligence*: Requiring information gathering from multiple sources, synthesis, and generation of reports. This workflow can adapt to query complexity and scale from simple fact-finding to comprehensive analysis.
- *Software Development*: Coding tasks where the *Orchestrator* can dynamically determine which files need modification, and coordinate specialized sub-agents for code generation, testing, debugging, and documentation.

### Considerations

- *Orchestration Effectiveness*: The orchestrator must handle complex task decomposition and worker selection. This can create bottlenecks and requires careful resource management, control loops, and error handling to avoid single points of failure in task distribution and orchestrator-worker communication.
- *Complexity Management*: Production systems often require hundreds of conversation turns, necessitating context compression, external memory systems, and observability mechanisms.
- *Evaluation Challenges*: Multi-agent systems require specialized evaluation approaches, including small-scale testing for rapid iteration and LLM-as-a-judge evaluation for complex requests.

## References

- [1] Balachandran, et al. 2025. *Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead*. [arxiv:2504.00294](https://arxiv.org/abs/2504.00294/).
- [2] Madaan, et al. 2023. *Self-Refine: Iterative Refinement with Self-Feedback*. [arxiv:2303.17651](https://arxiv.org/abs/2303.17651/).
- [3] Shinn, et al. 2023. *Reflexion: Language Agents with Verbal Reinforcement Learning*. [arxiv:2303.11366](https://arxiv.org/abs/2303.11366/)
- [4] Gou, et al. 2024. *CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing*. [arxiv:2305.11738](https://arxiv.org/abs/2305.11738/)
- [5] Patil, et al. 2023. *Gorilla: Large Language Model Connected with Massive APIs*. [arxiv:2305.15334](https://arxiv.org/abs/2305.15334/)
- [6] Yang, et al. 2023. *MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action*. [arxiv:2303.11381](https://arxiv.org/abs/2303.11381/)
- [7] Gao, et al. 2024. *Efficient Tool Use with Chain-of-Abstraction Reasoning*. [arxiv:2401.17464](https://arxiv.org/abs/2401.17464/)
- [8] Anthropic. 2024. *Model Context Protocol*. [anthropic:mcp](https://modelcontextprotocol.io/)
- [9] Wu, et al. 2022. *AI Chains: Transparent and Controllable Human-AI Interaction by Chaining Large Language Model Prompts*. [dl.acm:3491102.3517582](https://dl.acm.org/doi/abs/10.1145/3491102.3517582/)
- [10] Anthropic. 2024. *Building effective agents*. [anthropic:agents](https://www.anthropic.com/engineering/building-effective-agents/)
- [11] Cheng, et al. 2024. *Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects*. [arxiv:2401.03428](https://arxiv.org/abs/2401.03428/)
- [12] Anthropic. 2025. *How we built our multi-agent research system*. [anthropic:multi-agent-research](https://www.anthropic.com/engineering/built-multi-agent-research-system/)
