import { useState, useRef } from 'react'
import { Flame, Droplets, Wind, UploadCloud, Loader2, AlertCircle } from 'lucide-react'
import axios from 'axios'
import './index.css'

function App() {
  const [activeTab, setActiveTab] = useState('wildfire')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  
  const fileInputRef = useRef(null)

  const tabs = [
    { id: 'wildfire', name: 'Wildfire Prediction', icon: Flame },
    { id: 'waterbody', name: 'Water Body Segmentation', icon: Droplets },
    { id: 'spread', name: 'Wildfire Spread', icon: Wind },
  ]

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (selectedFile) => {
    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
    setResult(null)
    setError(null)
  }

  const handlePredict = async () => {
    if (!file) return

    setLoading(true)
    setError(null)
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      const endpoint = activeTab === 'waterbody' ? 'waterbody' : 'wildfire';
      const response = await axios.post(`http://localhost:8000/api/predict/${endpoint}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      setResult(response.data)
    } catch (err) {
      console.error(err)
      setError('Failed to get prediction from the backend. Make sure the server is running.')
    } finally {
      setLoading(false)
    }
  }

  const ActiveIcon = tabs.find(t => t.id === activeTab)?.icon || Flame;

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="app-title">
          <Flame size={28} color="#ff6b6b" />
          GeoPredict AI
        </div>
        
        <div className="nav-items">
          {tabs.map(tab => {
            const Icon = tab.icon
            return (
              <div 
                key={tab.id}
                className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => {
                  setActiveTab(tab.id);
                  setFile(null);
                  setPreview(null);
                  setResult(null);
                  setError(null);
                }}
              >
                <Icon size={20} />
                {tab.name}
              </div>
            )
          })}
        </div>
      </div>

      <div className="main-content">
        <div className="upload-card">
          {activeTab === 'spread' ? (
            <div style={{ color: 'var(--text-muted)' }}>
              <ActiveIcon size={48} style={{ marginBottom: '1rem', opacity: 0.5, display: 'inline-block' }} />
              <h2>Model Coming Soon</h2>
              <p style={{ marginTop: '0.5rem' }}>This prediction model is currently under development.</p>
            </div>
          ) : (
            <>
              {!preview ? (
                <div 
                  className="dropzone"
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <UploadCloud size={48} className="dropzone-icon" />
                  <h3 className="dropzone-text">Drag & Drop Satellite Image</h3>
                  <p className="dropzone-subtext">or click to browse standard formats (JPG, PNG)</p>
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    onChange={handleFileChange} 
                    accept="image/*" 
                    style={{ display: 'none' }} 
                  />
                </div>
              ) : (
                <div style={{ marginBottom: '2rem' }}>
                  {result && result.mask_base64 ? (
                    <div style={{ display: 'flex', gap: '2rem', justifyContent: 'center', marginBottom: '1rem', flexWrap: 'wrap' }}>
                      <div style={{ flex: '1', minWidth: '250px' }}>
                        <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-muted)', textAlign: 'center' }}>Original Image</h4>
                        <img src={preview} alt="Original" className="preview-image" style={{ width: '100%', display: 'block', margin: 0 }} />
                      </div>
                      <div style={{ flex: '1', minWidth: '250px' }}>
                        <h4 style={{ marginBottom: '0.5rem', color: '#4dabf7', textAlign: 'center' }}>Segmentation Mask</h4>
                        <div style={{ position: 'relative', width: '100%' }}>
                          <img src={preview} alt="Preview Base" className="preview-image" style={{ width: '100%', display: 'block', margin: 0 }} />
                          <img src={`data:image/png;base64,${result.mask_base64}`} alt="Mask overlay" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'fill', borderRadius: '12px', pointerEvents: 'none' }} />
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div style={{ position: 'relative', display: 'inline-block', marginBottom: '1rem' }}>
                      <img src={preview} alt="Preview" className="preview-image" style={{ display: 'block', margin: 0 }} />
                    </div>
                  )}
                  <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                    <button 
                      className="btn" 
                      style={{ background: 'transparent', border: '1px solid var(--border-color)', boxShadow: 'none' }}
                      onClick={() => { setFile(null); setPreview(null); setResult(null); }}
                    >
                      Remove
                    </button>
                    <button 
                      className="btn" 
                      onClick={handlePredict}
                      disabled={loading}
                      style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                    >
                      {loading ? <Loader2 size={18} className="spin" style={{ animation: 'spin 1s linear infinite' }} /> : <Flame size={18} />}
                      {loading ? 'Analyzing...' : 'Run Prediction'}
                    </button>
                  </div>
                </div>
              )}

              {error && (
                <div style={{ color: '#ff6b6b', background: 'rgba(255, 107, 107, 0.1)', padding: '1rem', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '0.5rem', textAlign: 'left', marginTop: '1rem' }}>
                  <AlertCircle size={20} />
                  {error}
                </div>
              )}

              {result && activeTab === 'waterbody' && (
                <div className="result-container">
                  <h3 className="result-title" style={{ color: '#4dabf7' }}>
                    Prediction: {result.prediction}
                  </h3>
                  <p style={{ color: 'var(--text-muted)', fontSize: '1rem', marginBottom: '1.5rem', lineHeight: '1.4' }}>
                    The model has segmented the satellite image and highlighted detected water bodies in yellow/cyan.
                  </p>
                </div>
              )}

              {result && activeTab === 'wildfire' && (
                <div className="result-container">
                  <h3 className="result-title" style={{ color: result.prediction === 'Wildfire' ? '#ff6b6b' : '#4dabf7' }}>
                    Prediction: {result.prediction}
                  </h3>
                  <p style={{ color: 'var(--text-muted)', fontSize: '1rem', marginBottom: '1.5rem', lineHeight: '1.4' }}>
                    The model predicts that the area in the given satellite image <strong style={{ color: result.prediction === 'Wildfire' ? '#ff6b6b' : '#4dabf7' }}>has {result.prediction === 'No Wildfire' ? 'not ' : ''}been affected</strong> by a wildfire.
                  </p>
                  
                  <div className="prob-bar-container">
                    <div className="prob-label">
                      <span>Wildfire Probability</span>
                      <span>{(result.wildfire_prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="prob-bar-bg">
                      <div 
                        className="prob-bar-fill fill-wildfire" 
                        style={{ width: `${result.wildfire_prob * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="prob-bar-container">
                    <div className="prob-label">
                      <span>No Wildfire Probability</span>
                      <span>{(result.no_wildfire_prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="prob-bar-bg">
                      <div 
                        className="prob-bar-fill fill-nowildfire" 
                        style={{ width: `${result.no_wildfire_prob * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
      <style>{`
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}

export default App
