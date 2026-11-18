# 🦜🕸️ LangGraph

[![PyPI - 版本](https://img.shields.io/pypi/v/langgraph?label=%20)](https://pypi.org/project/langgraph/#history)
[![PyPI - 许可证](https://img.shields.io/pypi/l/langgraph)](https://opensource.org/licenses/MIT)
[![PyPI - 下载量](https://img.shields.io/pepy/dt/langgraph)](https://pypistats.org/packages/langgraph)
[![未关闭的议题](https://img.shields.io/github/issues-raw/langchain-ai/langgraph)](https://github.com/langchain-ai/langgraph/issues)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain_oss)

需要 JS/TS 版本？请查看 [LangGraph.js](https://github.com/langchain-ai/langgraphjs)。

想更快地将 LangGraph 应用部署到生产环境？可以了解 [LangSmith](https://www.langchain.com/langsmith)。
[LangSmith](https://www.langchain.com/langsmith) 是一个统一的开发者平台，用于构建、测试和监控大语言模型（LLM）应用。

## 快速安装

```bash
uv add langgraph
```

## 🤔 LangGraph 是什么？

LangGraph 是一个底层编排框架，用于构建、管理和部署长时间运行、有状态的智能体。它提供持久执行、流式输出、人工介入、状态持久化和记忆等基础能力。

如果你的复杂需求涉及确定性工作流与智能体工作流的组合、深度定制以及对延迟的精细控制，建议使用 LangGraph。如果你希望借助预构建的智能体架构和模型集成，快速开发由大语言模型驱动的智能体与应用，可以使用 [LangChain](https://docs.langchain.com/oss/python/langchain/overview)。

LangChain 的[智能体](https://docs.langchain.com/oss/python/langchain/agents)构建在 LangGraph 之上，因此具备持久执行、流式输出、人工介入和状态持久化等能力。使用 LangChain 智能体的基本功能，无需先学习 LangGraph。

Klarna、Replit、Elastic 等推动智能体领域发展的企业已采用 LangGraph，用于大规模交付 AI 应用。

## 📖 文档

完整接口文档请参阅 [API 参考](https://reference.langchain.com/python/langgraph/)。概念指南、教程和使用示例请查看 [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/overview)。初次使用可以从[快速入门](https://docs.langchain.com/oss/python/langgraph/quickstart)开始。

## 📕 发布与版本管理

请参阅[发布政策](https://docs.langchain.com/oss/python/release-policy)和[版本管理政策](https://docs.langchain.com/oss/python/versioning)。

## 💁 参与贡献

作为快速发展领域中的开源项目，我们欢迎各种形式的贡献，包括新功能、基础设施改进和文档完善。

具体参与方式请参阅[贡献指南](https://docs.langchain.com/oss/python/contributing/overview)。

## 致谢

LangGraph 的设计受到 [Pregel](https://research.google/pubs/pub37252/) 和 [Apache Beam](https://beam.apache.org/) 的启发，公开接口借鉴了 [NetworkX](https://networkx.org/documentation/latest/) 的设计。LangGraph 由 LangChain 的开发团队 LangChain Inc 构建，也可以独立于 LangChain 使用。
