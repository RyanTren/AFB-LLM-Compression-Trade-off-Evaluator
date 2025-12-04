// import React, { useState } from 'react'
import ksuLogo from './assets/ksu.svg'
import TeamSection from "./components/team";
import './App.css'

function App() {

  return (
    <>
      <div>
        <a target="_blank">
          <img style={{height: 250}}src={ksuLogo} className="logo ksu logo-spin" alt="KSU logo" />
        </a>
      </div>

      <div style={{padding: '2px', textAlign: 'center'}}>
        <h1 style={{textShadow: '0 0 5px yelow'}}>P10-T1 LLM Comp Final Report</h1>
        <h3>developed & documented by</h3>
        <TeamSection />
      </div>
      
      <div className="info" style={{textShadow: '0 0 2px black', margin: '10px', padding: '10px'}}>
        <p>This project explores the trade-offs between various compression techniques applied to Large Language Models (LLMs). By evaluating different methods, we aim to provide insights into how compression affects model performance, efficiency, and resource utilization. Our findings are documented in the final report, which includes experimental results, analysis, and recommendations for practitioners looking to optimize LLMs for deployment in resource-constrained environments.

        </p>
      </div>

      <div className="card" style={{alignContent: 'center', display: 'column', gap: '5px', margin: '10px', padding: '10px'}}>
        <button onClick={() => window.open("/P10-T1-LLM-Comp-FinalReport-Draft.pdf", "_blank")}>
          Final Report PDF
        </button>
      </div>

      <div className="card" style={{alignContent: 'center', display: 'column', gap: '5px', margin: '10px', padding: '10px'}}>
        <button onClick={()=> window.open("https://github.com/RyanTren/AFB-LLM-Compression-Trade-off-Evaluator", "_blank")}>
          GitHub Repo
        </button>
      </div>

      <div className="card" style={{alignContent: 'center', display: 'column', gap: '5px', margin: '10px', padding: '10px'}}>
        <button onClick={() => window.open("https://youtube.com/watch?v=dQw4w9WgXcQ", "_blank")}>
          Final Presentation Video
        </button>
      </div>

    </>
  )
}

export default App
