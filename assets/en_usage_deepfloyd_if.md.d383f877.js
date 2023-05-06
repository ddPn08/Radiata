import{_ as s,c as a,o as e,R as l}from"./chunks/framework.f1d5a83d.js";const D=JSON.parse('{"title":"Deepfloyd IF","description":"","frontmatter":{},"headers":[],"relativePath":"en/usage/deepfloyd_if.md","lastUpdated":1683357224000}'),o={name:"en/usage/deepfloyd_if.md"},n=l(`<h1 id="deepfloyd-if" tabindex="-1">Deepfloyd IF <a class="header-anchor" href="#deepfloyd-if" aria-label="Permalink to &quot;Deepfloyd IF&quot;">​</a></h1><p>IF is a new image generation AI technology developed by the Deepfloyd team at Stability AI.</p><h2 id="usage" tabindex="-1">Usage <a class="header-anchor" href="#usage" aria-label="Permalink to &quot;Usage&quot;">​</a></h2><h3 id="windows" tabindex="-1">Windows <a class="header-anchor" href="#windows" aria-label="Permalink to &quot;Windows&quot;">​</a></h3><ol><li>Rewrite <code>webui-user.bat</code> as follows</li></ol><div class="language-bat"><button title="Copy Code" class="copy"></button><span class="lang">bat</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#89DDFF;">@echo</span><span style="color:#A6ACCD;"> </span><span style="color:#F78C6C;">off</span></span>
<span class="line"></span>
<span class="line"><span style="color:#89DDFF;">set</span><span style="color:#A6ACCD;"> PYTHON</span><span style="color:#89DDFF;">=</span></span>
<span class="line"><span style="color:#89DDFF;">set</span><span style="color:#A6ACCD;"> GIT</span><span style="color:#89DDFF;">=</span></span>
<span class="line"><span style="color:#89DDFF;">set</span><span style="color:#A6ACCD;"> VENV_DIR</span><span style="color:#89DDFF;">=</span></span>
<span class="line"><span style="color:#89DDFF;">set</span><span style="color:#A6ACCD;"> COMMANDLINE_ARGS</span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;">--deepfloyd_if</span></span>
<span class="line"></span>
<span class="line"><span style="color:#89DDFF;font-style:italic;">call</span><span style="color:#A6ACCD;"> launch.bat</span></span></code></pre></div><ol start="2"><li>Run <code>launch-user.bat</code></li></ol><h3 id="linux-or-macos" tabindex="-1">Linux or MacOS <a class="header-anchor" href="#linux-or-macos" aria-label="Permalink to &quot;Linux or MacOS&quot;">​</a></h3><ol><li>Rewrite <code>webui-user.sh</code> as follows</li></ol><div class="language-sh"><button title="Copy Code" class="copy"></button><span class="lang">sh</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#676E95;font-style:italic;"># export COMMANDLINE_ARGS=&quot;&quot;</span></span></code></pre></div><p>↓</p><div class="language-sh"><button title="Copy Code" class="copy"></button><span class="lang">sh</span><pre class="shiki material-theme-palenight"><code><span class="line"><span style="color:#C792EA;">export</span><span style="color:#A6ACCD;"> COMMANDLINE_ARGS</span><span style="color:#89DDFF;">=</span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">--deepfloyd_if</span><span style="color:#89DDFF;">&quot;</span></span></code></pre></div><ol start="2"><li>Run <code>launch-user.sh</code></li></ol>`,13),t=[n];function p(c,i,r,d,y,h){return e(),a("div",null,t)}const _=s(o,[["render",p]]);export{D as __pageData,_ as default};