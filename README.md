<div align="center">
  <a href="https://www.langchain.com/langgraph">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-light.svg">
      <img alt="LangGraph 标志" src=".github/images/logo-dark.svg" width="50%">
    </picture>
  </a>
</div>

<div align="center">
  <h3>用于构建有状态智能体的底层编排框架。</h3>
</div>

<div align="center">
  <a href="https://opensource.org/licenses/MIT" target="_blank"><img src="https://img.shields.io/pypi/l/langgraph" alt="PyPI - 许可证"></a>
  <a href="https://pypistats.org/packages/langgraph" target="_blank"><img src="https://img.shields.io/pepy/dt/langgraph" alt="PyPI - 下载量"></a>
  <a href="https://pypi.org/project/langgraph/" target="_blank"><img src="https://img.shields.io/pypi/v/langgraph.svg?label=%20" alt="版本"></a>
  <a href="https://x.com/langchain_oss" target="_blank"><img src="https://img.shields.io/twitter/url/https/twitter.com/langchain_oss.svg?style=social&label=Follow%20%40LangChain" alt="Twitter / X"></a>
</div>

<br>

LangGraph 是一个用于构建、管理和部署长时间运行的有状态智能体的底层编排框架，已被 Klarna、Replit、Elastic 等推动智能体应用发展的企业采用。

```bash
pip install -U langgraph
```

> [!TIP]
> 如果你希望快速构建智能体，可以了解 **[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)**。它是基于 LangGraph 构建的上层开发包，支持任务规划、子智能体和文件系统操作，适合处理复杂任务。

如果你需要对应的 JavaScript / TypeScript 库，请参阅 [LangGraph.js](https://github.com/langchain-ai/langgraphjs) 和 [JavaScript 文档](https://docs.langchain.com/oss/javascript/langgraph/overview)。

## 为什么使用 LangGraph？

LangGraph 为各类长时间运行的有状态工作流和智能体提供底层基础设施：

- **[持久执行](https://docs.langchain.com/oss/python/langgraph/durable-execution)**：构建能够长时间运行、在故障后保留进度，并从中断位置自动恢复执行的智能体。
- **[人工介入](https://docs.langchain.com/oss/python/langgraph/interrupts)**：支持在执行过程中检查和修改智能体状态，将人工监督融入工作流。
- **[完整的记忆能力](https://docs.langchain.com/oss/python/langgraph/memory)**：同时支持用于当前推理过程的短期工作记忆，以及跨会话保存的长期持久记忆。
- **[使用 LangSmith 调试](https://www.langchain.com/langsmith)**：通过可视化工具追踪执行路径、查看状态变化并获取详细的运行指标，深入理解复杂的智能体行为。
- **[面向生产环境的部署](https://docs.langchain.com/langsmith/deployments)**：使用可扩展的基础设施部署智能体系统，应对有状态、长时间运行的工作流所带来的挑战。

> [!TIP]
> 如需开发、调试和部署 AI 智能体及大语言模型应用，请参阅 [LangSmith](https://docs.langchain.com/langsmith/home)。

## LangGraph 生态

LangGraph 可以独立使用，也能与 LangChain 旗下的其他产品配合，为开发者提供完整的智能体开发工具。

你可以将 LangGraph 与以下产品结合使用：

- [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)：构建能够规划任务、使用子智能体并利用文件系统处理复杂任务的智能体。
- [LangChain](https://docs.langchain.com/oss/python/langchain/overview)：提供丰富的集成和可组合组件，简化大语言模型应用开发。
- [LangSmith](https://www.langchain.com/langsmith)：用于智能体评估与可观测性。调试表现不佳的大语言模型调用，评估智能体执行轨迹，观察生产环境行为，并持续改进性能。
- [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments)：使用专为长时间运行的有状态工作流构建的部署平台，部署和扩展智能体。在团队之间发现、复用、配置和共享智能体，并通过 [LangSmith Studio](https://docs.langchain.com/langsmith/studio) 的可视化原型工具快速迭代。

---

## 文档

- [docs.langchain.com](https://docs.langchain.com/oss/python/langgraph/overview)：完整文档，包含概念介绍与使用指南。
- [reference.langchain.com/python/langgraph](https://reference.langchain.com/python/langgraph)：LangGraph 各开发包的 API 参考文档。
- [LangGraph 快速入门](https://docs.langchain.com/oss/python/langgraph/quickstart)：开始使用 LangGraph 构建智能体。
- [Chat LangChain](https://chat.langchain.com/)：通过对话查询 LangChain 文档并获取问题解答。

**社区讨论**：访问 [LangChain 论坛](https://forum.langchain.com)，与社区交流技术问题、想法和反馈。

## 更多资源

- **[开发指南](https://docs.langchain.com/oss/python/learn)**：围绕流式输出、记忆与持久化、分支和子图等主题，提供简明实用的代码示例。
- **[LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph)**：通过免费的系统课程学习 LangGraph 基础。
- **[应用案例](https://www.langchain.com/built-with-langgraph)**：了解行业企业如何使用 LangGraph 大规模交付 AI 应用。
- [贡献指南](https://docs.langchain.com/oss/python/contributing/overview)：了解如何参与 LangChain 项目，并寻找适合入门的议题。
- [行为准则](https://github.com/langchain-ai/langchain/?tab=coc-ov-file)：社区参与规范与行为标准。

---

## 致谢

LangGraph 的设计受到 [Pregel](https://research.google/pubs/pub37252/) 和 [Apache Beam](https://beam.apache.org/) 的启发，其公共接口借鉴了 [NetworkX](https://networkx.org/documentation/latest/)。LangGraph 由 LangChain 的开发团队 LangChain Inc 构建，但也可以在不使用 LangChain 的情况下独立使用。
