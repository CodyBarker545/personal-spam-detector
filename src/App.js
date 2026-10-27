import React, { useState } from 'react';
import './App.css';

function App() {
  const [comment, setComment] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!comment.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comment }),
      });

      const data = await response.json();
      setResult(data.prediction);
    } catch (err) {
      setResult('Error contacting server.');
      console.error(err);
    }

    setLoading(false);
  };

  return (
    <div className="app-container">
      <h1>YouTube Spam Comment Detector</h1>
      <textarea
        rows="6"
        placeholder="Enter your comment here..."
        value={comment}
        onChange={(e) => setComment(e.target.value)}
      />
      <button onClick={handlePredict}>
        {loading ? 'Analyzing...' : 'Check for Spam'}
      </button>
      {result && (
        <div className={`result ${result === 'Spam' ? 'spam' : 'not-spam'}`}>
          Result: {result}
        </div>
      )}
    </div>
  );
}

export default App;
