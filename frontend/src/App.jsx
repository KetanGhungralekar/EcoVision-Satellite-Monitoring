import { useState, useRef } from 'react'
import { Flame, Droplets, Map, Trees, UploadCloud, Loader2, AlertCircle } from 'lucide-react'
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
    { id: 'wildfire',  name: 'Wildfire Prediction',       icon: Flame,    accept: 'image/*', hint: 'JPG, PNG' },
    { id: 'waterbody', name: 'Water Body Segmentation',   icon: Droplets, accept: 'image/*', hint: 'JPG, PNG' },
    { id: 'burnscar',  name: 'Burned Area Segmentation',  icon: Map,      accept: '.tif,.tiff', hint: 'GeoTIFF (.tif)' },
    { id: 'deforestation', name: 'Deforestation Detection', icon: Trees, accept: 'image/*', hint: 'JPG, PNG' },
  ]

  const activeTabInfo = tabs.find(t => t.id === activeTab)

  const handleDragOver = (e) => e.preventDefault()

  const handleDrop = (e) => {
    e.preventDefault()
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0])
  }

  const handleFileChange = (e) => {
    if (e.target.files?.[0]) handleFile(e.target.files[0])
  }

  const handleFile = (selectedFile) => {
    setFile(selectedFile)
    // GeoTIFFs can't be previewed by the browser — we'll get rgb_base64 from backend
    if (activeTab === 'burnscar') {
      setPreview(null)
    } else {
      setPreview(URL.createObjectURL(selectedFile))
    }
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
      const response = await axios.post(
        `http://localhost:8000/api/predict/${activeTab}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      setResult(response.data)
    } catch (err) {
      const backendErr = err.response?.data?.error || err.response?.data?.detail
      setError(backendErr || 'Failed to get prediction. Make sure the server is running.')
    } finally {
      setLoading(false)
    }
  }

  const resetState = () => {
    setFile(null); setPreview(null); setResult(null); setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  // Resolve the image src for the "original" pane
  // For burnscar the backend returns rgb_base64 because TIFF can't be shown by browser
  const originalSrc = activeTab === 'burnscar' && result?.rgb_base64
    ? `data:image/png;base64,${result.rgb_base64}`
    : preview

  const hasMask = result?.mask_base64

  return (
    <div className="app-container">
      {/* ── Sidebar ──────────────────────────────────────────────── */}
      <div className="sidebar">
        <div className="app-title">
          <Flame size={28} color="#ff6b6b" />
          EcoVision AI
        </div>

        <div className="nav-items">
          {tabs.map(tab => {
            const Icon = tab.icon
            return (
              <div
                key={tab.id}
                className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => { setActiveTab(tab.id); resetState() }}
              >
                <Icon size={20} />
                {tab.name}
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Main ─────────────────────────────────────────────────── */}
      <div className="main-content">
        <div className="upload-card">

          {/* Dropzone — shown when no file selected */}
          {!file ? (
            <div
              className="dropzone"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <UploadCloud size={48} className="dropzone-icon" />
              <h3 className="dropzone-text">Drag &amp; Drop Satellite Image</h3>
              <p className="dropzone-subtext">
                or click to browse &nbsp;·&nbsp;
                <span style={{ color: activeTab === 'burnscar' ? '#ff9f43' : '#4dabf7' }}>
                  {activeTabInfo?.hint}
                </span>
              </p>
              {activeTab === 'burnscar' && (
                <p style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)', opacity: 0.7 }}>
                  Requires a 6-band Sentinel-2 / HLS GeoTIFF (B02–B07)
                </p>
              )}
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept={activeTabInfo?.accept}
                style={{ display: 'none' }}
              />
            </div>
          ) : (
            /* ── Preview / Result area ────────────────────────── */
            <div style={{ marginBottom: '2rem' }}>
              {/* Before prediction: show a placeholder for TIFF, or the image preview */}
              {!result ? (
                <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
                  {preview ? (
                    <img src={preview} alt="Preview" className="preview-image" />
                  ) : (
                    <div style={{
                      padding: '2rem', borderRadius: '12px',
                      background: 'rgba(255,159,67,0.08)',
                      border: '1px dashed rgba(255,159,67,0.4)',
                      color: '#ff9f43'
                    }}>
                      <Map size={40} style={{ marginBottom: '0.5rem', opacity: 0.8 }} />
                      <p style={{ margin: 0, fontWeight: 600 }}>{file.name}</p>
                      <p style={{ margin: '0.25rem 0 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                        GeoTIFF ready · click <strong>Run Prediction</strong> to process
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                /* After prediction */
                hasMask ? (
                  <div style={{ display: 'flex', gap: '2rem', justifyContent: 'center', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                    {/* Original */}
                    <div style={{ flex: '1', minWidth: '250px' }}>
                      <h4 style={{ marginBottom: '0.5rem', color: 'var(--text-muted)', textAlign: 'center' }}>Original Image</h4>
                      {originalSrc
                        ? <img src={originalSrc} alt="Original" className="preview-image" style={{ width: '100%', display: 'block', margin: 0 }} />
                        : <div style={{ padding: '3rem', background: 'rgba(255,255,255,0.03)', borderRadius: '12px', color: 'var(--text-muted)', textAlign: 'center' }}>No RGB preview</div>
                      }
                    </div>
                    {/* Mask overlay */}
                    <div style={{ flex: '1', minWidth: '250px' }}>
                      <h4 style={{
                        marginBottom: '0.5rem', textAlign: 'center',
                        color: activeTab === 'burnscar' ? '#ff9f43' : (activeTab === 'deforestation' ? '#40c057' : '#4dabf7')
                      }}>
                        {activeTab === 'burnscar' 
                          ? 'Burn Scar Mask' 
                          : activeTab === 'deforestation' ? 'Deforestation Mask' : 'Segmentation Mask'}
                      </h4>
                      <div style={{ position: 'relative', width: '100%' }}>
                        {originalSrc
                          ? <img src={originalSrc} alt="Base" className="preview-image" style={{ width: '100%', display: 'block', margin: 0 }} />
                          : <div style={{ height: '224px', background: '#111', borderRadius: '12px' }} />
                        }
                        <img
                          src={`data:image/png;base64,${result.mask_base64}`}
                          alt="Mask"
                          style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'fill', borderRadius: '12px', pointerEvents: 'none' }}
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  /* Wildfire result — no mask, just show the image */
                  <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
                    <img src={originalSrc || preview} alt="Preview" className="preview-image" />
                  </div>
                )
              )}

              {/* Action buttons */}
              <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                <button
                  className="btn"
                  style={{ background: 'transparent', border: '1px solid var(--border-color)', boxShadow: 'none' }}
                  onClick={resetState}
                >
                  Remove
                </button>
                <button
                  className="btn"
                  onClick={handlePredict}
                  disabled={loading}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '0.5rem',
                    background: activeTab === 'burnscar' ? 'linear-gradient(135deg, #ff9f43, #e55039)' 
                              : (activeTab === 'deforestation' ? 'linear-gradient(135deg, #40c057, #2b8a3e)' : undefined)
                  }}
                >
                  {loading
                    ? <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
                    : <Flame size={18} />}
                  {loading ? 'Analyzing…' : 'Run Prediction'}
                </button>
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div style={{ color: '#ff6b6b', background: 'rgba(255,107,107,0.1)', padding: '1rem', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '1rem' }}>
              <AlertCircle size={20} />
              {error}
            </div>
          )}

          {/* ── Result panels ──────────────────────────────────── */}

          {result && activeTab === 'wildfire' && (
            <div className="result-container">
              <h3 className="result-title" style={{ color: result.prediction === 'Wildfire' ? '#ff6b6b' : '#4dabf7' }}>
                Prediction: {result.prediction}
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '1rem', marginBottom: '1.5rem', lineHeight: '1.4' }}>
                The model predicts that the area in the given satellite image{' '}
                <strong style={{ color: result.prediction === 'Wildfire' ? '#ff6b6b' : '#4dabf7' }}>
                  has {result.prediction === 'No Wildfire' ? 'not ' : ''}been affected
                </strong>{' '}
                by a wildfire.
              </p>
              <div className="prob-bar-container">
                <div className="prob-label">
                  <span>Wildfire Probability</span>
                  <span>{(result.wildfire_prob * 100).toFixed(1)}%</span>
                </div>
                <div className="prob-bar-bg">
                  <div className="prob-bar-fill fill-wildfire" style={{ width: `${result.wildfire_prob * 100}%` }} />
                </div>
              </div>
              <div className="prob-bar-container">
                <div className="prob-label">
                  <span>No Wildfire Probability</span>
                  <span>{(result.no_wildfire_prob * 100).toFixed(1)}%</span>
                </div>
                <div className="prob-bar-bg">
                  <div className="prob-bar-fill fill-nowildfire" style={{ width: `${result.no_wildfire_prob * 100}%` }} />
                </div>
              </div>
            </div>
          )}

          {result && activeTab === 'waterbody' && (
            <div className="result-container">
              <h3 className="result-title" style={{ color: '#4dabf7' }}>
                Prediction: {result.prediction}
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '1rem', lineHeight: '1.4' }}>
                The model has segmented the satellite image and highlighted detected water bodies.
              </p>
            </div>
          )}

          {result && activeTab === 'burnscar' && (
            <div className="result-container">
              <h3 className="result-title" style={{ color: '#ff9f43' }}>
                {result.prediction_text}
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '1rem', lineHeight: '1.4' }}>
                The <strong style={{ color: '#ff9f43' }}>Prithvi-100M foundation model</strong> analysed the 6-band Sentinel-2 GeoTIFF.
                Pixels highlighted in <strong style={{ color: '#ff9f43' }}>orange</strong> are classified as burned / post-fire scar areas.
              </p>
            </div>
          )}

          {result && activeTab === 'deforestation' && (
            <div className="result-container">
              <h3 className="result-title" style={{ color: '#40c057' }}>
                {result.prediction_text}
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '1rem', lineHeight: '1.4' }}>
                The <strong style={{ color: '#40c057' }}>U-Net segmentation model</strong> has highlighted areas of forest cover and deforestation.
                <br />
                <span style={{ display: 'inline-block', width: '12px', height: '12px', background: '#40c057', borderRadius: '2px', marginRight: '6px' }}></span>
                <strong style={{ color: '#40c057' }}>Green</strong>: Forest &nbsp;
                <span style={{ display: 'inline-block', width: '12px', height: '12px', background: '#ff6b6b', borderRadius: '2px', marginRight: '6px', marginLeft: '12px' }}></span>
                <strong style={{ color: '#ff6b6b' }}>Red</strong>: Deforested area
              </p>
            </div>
          )}
        </div>
      </div>

      <style>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
    </div>
  )
}

export default App
