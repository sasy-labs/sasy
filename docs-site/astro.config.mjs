import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLlmsTxt from 'starlight-llms-txt';
export default defineConfig({
  site: 'https://docs.sasy.ai',
  redirects: { '/first-agent/': '/examples/' },
  integrations: [starlight({ title: 'SASY',
  customCss: ['./src/styles/layout.css'],
  // Serves /llms.txt, /llms-full.txt and /llms-small.txt for coding agents.
  plugins: [starlightLlmsTxt({
    description: 'SASY is a policy enforcement engine for AI agents. A Python SDK '
      + 'instruments the agent; a Rust engine, run locally with `sasy engine start`, '
      + 'checks each tool call against a Datalog policy over the history behind it.',
  })],
  sidebar: [
    { label: 'Start here', items: [
      { label: 'Get started', slug: 'get-started' },
      { label: 'SDK usage patterns', slug: 'sdk' },
      { label: 'Examples', slug: 'examples' },
      { label: 'The dependency graph', slug: 'dependency-graph' },
      { label: 'System architecture', slug: 'architecture' },
      { label: 'Concepts', slug: 'concepts' },
    ] },
    { label: 'Write policies', items: [
      { label: 'Policy language', slug: 'policy-language' },
    ] },
    { label: 'Integrate a framework', items: [
      { label: 'Use the SDK directly', slug: 'quickstart' },
      { label: 'LangChain', slug: 'integrations/langchain' },
      { label: 'Google ADK', slug: 'integrations/google-adk' },
      { label: 'Langroid', slug: 'integrations/langroid' },
      { label: 'Add a framework', slug: 'instrumentation' },
    ] },
    { label: 'Adapter reference', items: [
      { label: 'LangChain records', slug: 'reference/langchain-records' },
      { label: 'ADK records', slug: 'reference/google-adk-records' },
    ] },
    { label: 'Operate', items: [
      { label: 'Build from source', slug: 'building' },
      { label: 'Authentication and roles', slug: 'authentication' },
      { label: 'Configuration', slug: 'configuration' },
      { label: 'Engine CLI', slug: 'cli' },
      { label: 'Engine environment', slug: 'environment' },
      { label: 'Security model and limits', slug: 'limits' },
    ] },
  ] })],
});
