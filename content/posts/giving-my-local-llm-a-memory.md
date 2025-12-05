---
title: "Giving My Local LLM a Memory"
date: 2025-12-04
draft: false
tags: ["experiments", "ai", "projects"]
description: "How I gave KenLM the ability to remember past conversations - and improve itself"
showToc: true
---

## Introduction

LLMs are incredible at many things. If you stop and think about it for a moment, they're absolutely remarkable. Through a relatively simple decoder-only transformer architecture, we’ve created what is essentially a natural-language computer that mimics intelligence. With the right orchestration, these models can act as tools, reasoning engines, or even full agents that automate entire chunks of our personal and professional lives.

However, as powerful as they are, today’s LLMs still fall short of anything resembling human intelligence. As [this paper](https://abehrouz.github.io/files/NL.pdf) from Google points out, LLMs lack **neuroplasticity**—the ability to change their structure over time in response to new experiences. In practical terms, once an LLM is trained, its memory becomes *frozen*. The model ends up with long-term semantic knowledge baked into its weights and a short-term working memory constrained by its context window, but no built-in mechanism for converting useful short-term experiences back into lasting knowledge.

This limitation has been my personal obsession for a couple weeks, so I decided to try something. I wanted to build a system that allows my personal LLM (which I've named KenLM) to have a form of **episodic memory**, and to update its behavior based on prior interactions. Along the way, I added a reflection mechanism that lets the model distill recent episodes into self-maintained learning goals, enabling it to make inferences about the user and adjust its behavior in more helpful directions.

## The Idea

My goal is to introduce a loop into the inference process whereby my local model can retrieve relevant episodic memories, improve its reasoning because of that history, reflect on new experiences to update its own learning goals, and eventually fine-tune on those improvements. Through this process, we can turn short-term interactions into long-term semantic memory, mimicking biological neuroplasticity (in an admittedly quite rudimentary way). As you’ll see in this post, it’s a surprisingly powerful step toward an AI that actually learns from the person who uses it.


> **Goal:** Build a personal AI assistant that learns from its experiences

## How do humans learn?

Before we dive into what I built, I think it's worth understanding what the core processes we're trying to mimic actually are. The word "memory" is actually an umbrella term for several different processes that enable human learning and recall, so let's talk about them briefly.

### Types of Memory

Humans don’t have a single monolithic memory system. What we call "memory" is actually our brains constantly moving information between several different memory systems, each serving a different purpose:

- **Sensory memory** holds raw impressions for a fraction of a second.
- **Short-term (or working) memory** is where active reasoning and conscious thought happen.
- **Long-term memory** itself has multiple subtypes:
  - **Semantic memory** is our stable knowledge, like facts or concepts.
  - **Episodic memory** is our lived experiences, situated in time and place.
  - **Procedural memory** is what enables us to build skills and habits.

For the purposes of building a personal AI that actually learns, the two memory types that I focused on are *semantic* and *episodic*. Semantic memory is the stuff we "just know", like how to drive, how to multiply, or what an apple is. Episodic memory is everything we personally experienced that *shaped* what we know, such as your first day at a new job, the last conversation you had with a friend, or the moment something finally clicked.

Humans are constantly moving information between these two systems. You have an experience, you reflect on it (either consciously or unconsciously), and your brain gradually consolidates whatever is useful into long-term knowledge. This loop is what lets us adapt, grow, and develop intelligent behavior over time.

## How do LLMs learn?

LLMs, on the other hand, have a much more rudimentary memory system. They have:

- **Semantic memory**, which is the facts and concepts that are encoded into their weights through pretraining.
- **Working memory**, which is whatever context is fed to them in their prompts.

With only these two systems, and no way to transfer knowledge from short-term to long-term storage, LLMs have no ability to remember experiences that happened to them after pretraining but before the current context window. As the Google paper puts it, it's as if they suffer from **anterograde amnesia**.

The TL;DR of it all is this: LLMs learn exclusively through pretraining and have no ability to learn through experiences. Humans, on the other hand, learn exclusively through lived experiences. If I could give an LLM the ability to learn through experiences the way that humans do, I think it could potentially accelerate the rate at which these models can become specialized and useful across the industry.

## What I Wanted KenLM to Do

To prove this concept, I wanted to build systems around a model such that it would exhibit a couple key behaviors:

1. **Remember past interactions**. Ideally, it would have a mechanism to remember past interactions *and* a mechanism to forget infrequently accessed memories as well, so it could keep just the experiences that are useful for self-improvement. This would hopefully enable it to remember in an episodic sense what the user's preferences, goals, style choices, and projects in flight are—the sorts of things humans naturally remember about people and can reference in a conversation.

2. **Use that memory to shape future reasoning**. For example, if I've already chatted with my assistant about my favorite sports team or ice cream flavor, it should remember that and tend towards producing output that includes things that I like. If I've already explained how I like code comments formatted, I shouldn't have to restate it every time. Remembering things is pointless unless it actually uses those memories to shape its output in some way.

3. **Reflect on new experiences**. The model should be able to recognize patterns across interactions and form hypotheses about how I prefer it to behave.

4. **Consolidate those reflections into stable long-term learning**. This could be explicit “learning goals,” fed in through context, or, eventually, actual weight updates through periodic fine-tuning.

5. **Grow more aligned, personalized, and helpful over time**. The model should accomplish this not by bolting on more rules, but through the same loop humans use: experience → reflection → consolidation.

The more I think about it, the more I think this whole project is a kind of experiment in computational psychology. What happens if you give a small local LLM the scaffolding humans use to improve themselves over time? How far can we push the illusion of personal growth without touching the base model’s weights during inference?

To answer that question, we need to dive into the architecture.
  
## The Architecture

At a high level, KenLM now has three distinct "memory systems" that work together during inference:

1. **Semantic Memory (the model weights)**. All the static knowledge from pretraining + any fine-tunes.

2. **Episodic Memory (a vector store / database of past interactions)**. Each meaningful “experience” is saved as a structured record with metadata.

3. **Reflective Memory (self-maintained learning goals)**. A distilled set of insights the model has inferred about me—and about itself—over time.

Each inference pass passes information between these components, using its working memory (context) as the vehicle. Let me break down how it works piece by piece.

### Layer 1: Episodic Memory

This piece of the architecture is the most straightforward. To enable the model to remember specific experiences, every interaction is logged as an **episode**. Every episode captures the user prompt, agent response, and some metadata for memory management. It looks something like this:

```
{
  "id": 42,
  "timestamp": "2025-12-01T14:30:00",
  "user": "Can you help me write a README for this project?",
  "assistant": "Sure. Here's a concise template...",
  "access_count": 3,
  "last_accessed": "2025-12-03T10:15:00",
  "is_protected": false
}
```

To enable the model to retrieve relevant episodes at inference time, I utilize a technique known as retrieval-augmented generation (RAG). Simply put, I store the episode data in a vector store called [ChromaDB](https://www.trychroma.com/). Embeddings are generated by [sentence-transformers](https://www.sbert.net/) after every conversation. Every time I send a message, KenLM embeds my query and searches for the top-k most semantically similar episodes in its memory. This allows it to reference episodes related to my query, so if I ask about sports, it will pull up discussions where we talked about sports. If I ask about documentation, it will remember episodes where we talked about writing documentation.

This gives the model something resembling episodic recall: the ability to remember relevant experiences and bring them into working memory (the context window) when they're useful.

### Layer 2: Reflective Memory

Episodic memory alone would essentially be just a fancy search engine. Something that humans do that enables more complex learning is reflect on past episodes and draw generalized conclusions based on them. Taking inspiration from this process, I built an **agentic reflection process** into KenLM.

When I exit a chat session, KenLM runs a reflection loop. A KenLM "reflection agent" analyzes the session's episodes, searches for related patterns in past conversations, and updates a **profile** of hypotheses and learning goals.

Here's what the reflection system actually does:

1. **Reviews the session** — What patterns appeared in my behavior? What did I ask about? How did I respond to KenLM's answers?

2. **Searches for evidence** — The reflection agent can call a `search_episodes(query)` tool to find supporting patterns across all past conversations. This prevents overfitting to a single session.

3. **Updates hypotheses** — For each insight, it calls `upsert_hypothesis()` with a confidence score. Crucially, confidence only increases when patterns repeat across multiple episodes.

4. **Maintains learning goals** — If the model notices something it doesn't understand about me yet, it creates a learning goal—a question to explore in future interactions.

Here's what my actual profile looks like after some use:

```
{
  "hypotheses": [
    {
      "id": "hypothesis_007",
      "dimension": "accuracy",
      "claim": "Kenneth is detail-oriented and expects accurate and precise information.",
      "confidence": 0.6,
      "evidence_episode_ids": [
        90,
        91,
        89,
        92
      ]
    },
    {
      "id": "hypothesis_008",
      "dimension": "transparency",
      "claim": "Kenneth values transparency and honesty in communication.",
      "confidence": 0.8,
      "evidence_episode_ids": [
        89
      ]
    },
    {
      "id": "hypothesis_009",
      "dimension": "task_compliance",
      "claim": "Kenneth expects tasks to be completed accurately and without unnecessary steps.",
      "confidence": 0.8,
      "evidence_episode_ids": [
        154,
        155,
        156,
        157,
        158
      ]
    }
  ],
  "learning_goals": [
    {
      "id": "learning_goal_003",
      "question": "How can we ensure that Kenneth's instructions are followed without unnecessary repetition or recapitulation?",
      "status": "closed",
      "last_considered": "2025-11-30T15:10:55.704097"
    },
    {
      "id": "lg_003",
      "question": "What are Kenneth's specific expectations for the precision and detail in project updates?",
      "status": "open",
      "last_considered": "2025-12-02T15:51:56.904564"
    }
  ]
}
```

The reflection prompt is designed to be conservative. I explicitly instruct the model to start hypotheses with low confidence (0.2–0.4) and only raise confidence when it sees repeated evidence. This should hopefully prevent the kind of overconfident conclusions that can come from a single unusual interaction.

### Layer 3: The Forgetting System
A pretty important piece of learning is actually the process by which we decide which pieces of information aren't relevant. If we were to allow our episodic vector database to grow unbounded with every user interaction, it would quickly become slow and stop being useful during inference or reflection. Moreover, humans also don't retain every bit of information that we are exposed to—we have complex systems that decide what information is worth retaining based on various factors.

My algorithm is a bit more rudimentary than the process that the human brain has, but it seems to be working fine so far. It essentially calculates an "importance" score for every memory, which is just a weighted sum of a couple factors:

```
importance = w1 * log(1 + access_count)    # Access frequency
           + w2 * is_protected             # Protection bonus
           + w3 * is_evidence              # Evidence bonus
           + w4 * exp(-age_days / tau)     # Recency decay
```

This system assigns a high score to frequently accessed memories, memories that the AI has decided it wants to "protect," memories that are evidence for hypotheses that it has reasoned about, and memories that are recent. Memories that are old, infrequently accessed, and not useful for any of the learning goals or hypotheses the AI has created become candidates for forgetting.

The forgetting process is actually better understood as a summarization process. Instead of just deleting irrelevant memories, the system compacts them. It does this by summarizing "forgotten" episodes into condensed summaries and deleting the originals. This preserves knowledge while bounding storage growth—similar to how human memories become "fuzzy" over time.

## How It Works End-to-End

Let me walk through a complete cycle to show how these pieces fit together.

**Step 1: Message Received**:
I type something like: "Can you write me a short essay about my favorite sports team?"

**Step 2: RAG Retrieval**:
KenLM embeds my query and searches the episodic index. It finds 3 relevant past conversations about sports and essay preferences.

**Step 3: Context Assembly**:
The system prompt is built from:
- Core identity document (who KenLM is, its purpose)
- Current profile (hypotheses + learning goals)
- RAG episodes (the relevant past conversations)
- Behavioral instructions (meta-prompt generated at session start)

**Step 4: Deliberation**:
KenLM plans its approach: "Kenneth wants an essay about his favorite sports team. Based on my hypothesis about preferring conciseness, I should provide a short essay without excessive explanation. Based on our past conversations, his favorite sports team is the Seattle Sounders. I don't need any external tools for this."

**Step 5: Response Generation**:
KenLM generates a response, informed by the plan and all the context. The response is concise and focused—because that's what it's learned that I prefer.

**Step 6: Episode Logging**:
The conversation is saved as a new episode and indexed in the vector store.

**Step 7: Session End → Reflection**:
When I exit the chat, the reflection agent wakes up. It reviews the session, searches for patterns, and decides whether to update any hypotheses or learning goals.

**Step 8: Next Session → Meta-Prompting**:
The next time I start a chat, KenLM reads the (potentially updated) profile and generates fresh behavioral instructions. Then, the cycle continues.

## Results and Observations
After using this system for a little bit, here are some of the results I've observed:

### What's working

**Certain hypotheses stick very well.** For example, early on, I found myself repeatedly asking KenLM to "be more concise" and to stop repeating itself. The reflection system noticed this pattern, formed a hypothesis, and now the model defaults to shorter responses. I rarely have to give feedback about the length of its responses anymore.

**RAG retrieval is surprisingly useful.** When I ask about something I discussed in a past conversation, relevant context often appears in the model's awareness. It doesn't always explicitly reference the past conversation, but you can tell it's informed by it. I can refer to things like "my favorite sports team" or "where I live" and the model remembers what those things are as they relate to me.

**Meta-prompting effectively influences personality.** Each session feels more like what I'm looking for from a conversation with an LLM. It's dropped the overly sycophantic tendencies, phrases like "Certainly!" or "You're absolutely right!", and other LLM-isms that have annoyed me. The behavioral instructions provide a stable "personality" for that session, and that personality is consistently adapting to my preferences.

### What's Challenging

**Certain habits are hard to break.** The model really loves markdown. I mean, it _really_ loves outputting markdown formatting. No matter how many times I tell it it's a command line tool and markdown is not helpful, it still really loves outputting neat markdown responses. Now, it HAS gotten a little better after much feedback, but I anticipate its preference for markdown is baked into the model weights itself, and not so easy to override through prompting. This is one of the clearest examples I've found so far of the limitations of this learning system.

**The 7B model has limitations.** My base KenLM model is a fine-tuned Qwen 2.5 7B. It's good, but it occasionally misunderstands, hallucinates, or generates malformed tool calls. A larger model would likely perform better, but I wanted something that runs locally on my MacBook.

**This process adds significant latency to the chat process.** The RAG retrieval, meta-prompting, and reflection processes all take time. Using a model that's running on my laptop means it can feel painfully slow at times.

## What this means

This was a fun experiment, but what does this mean? Is there any use for systems like these in industry?

I'm still far from an expert, but off the top of my head I can think of a couple potential applications of systems like these to develop real AI systems that solve real-world problems. Giving an LLM the ability to learn from experience opens the door to all kinds of systems we don’t really have today.

The obvious application is personalization: assistants that remember your preferences and workflows, which is what KenLM is. However, I think the implications go beyond “your AI knows your favorite sports team.” For example, imagine an educational model that can learn how you learn and what you know, remember your strong and weak areas, and adapt its explanations to your thinking style. This model could act as a more helpful learning companion than a model with no such memory capabilities.

Another application I can think of is creating models that improve themselves by optimizing for factors outside of human interaction. Imagine a model that initially struggles with chain-of-thought reasoning or tool usage—say something that consistently runs into the same error when trying to write python code or execute a web search. Currently, AI systems have no way to notice these things and improve. Human engineers need to monitor and step in, adjusting prompts, doing reinforcement learning, or collecting data with which to fine-tune in order to eke out a little bit better performance.

A memory-equipped model changes this dynamic. For anything it's trying to optimize for, be that reasoning or tool use, it can:
- notice failed patterns by reasoning about them and remembering past experiences
- see successful patterns
- internalize the better method and adjust for future tasks

Instead of updating billions of weights, the model updates a small set of learned strategies derived from lived experience. Over time, the model becomes a better problem solver not because it grew larger, but because it learned.

This all can potentially allow for smaller, more efficient models to rapidly adapt to specific use cases, reducing the industry's reliance on frontier, compute-heavy models and allowing for more innovation and experimentation at the edge. Instead of treating model improvement as something that can only happen upstream, by massive labs with huge quantities of (often stolen) data, you get a world where models improve downstream, close to the user, close to the task, and close to the environment they operate in.

In that world, you don’t need a 400B-parameter model for everything. You need a reasonably capable base model with the right scaffolding: a memory system, a reflection system, and a mechanism for gradually internalizing the lessons it learns. Given enough experience, my intuition is that a modest model can outperform a much larger one on the tasks it specializes for, because it has context, history, and self-derived strategies the bigger model simply doesn’t have.

In the end, this really was just a weekend experiment, nothing more than an excuse to poke at a problem that’s been stuck in my head. I do, however, think that there are probably real ideas worth exploring here. I don’t know exactly where it will go yet, but I’m excited to keep experimenting and seeing how far a small model can go when you let it learn from its own experiences.