import React, { useState } from 'react';
import { Upload, Loader } from 'lucide-react';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleImageUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setIsLoading(true);
      setError(null);
      setResults(null);

      const formData = new FormData();
      formData.append('image', file);

      const response = await fetch('http://localhost:4000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process image');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">
          Breast Ultrasound Image Analysis
        </h1>

        {/* Upload Section */}
        <div className="bg-white p-8 rounded-lg shadow-md mb-8">
          <div className="flex flex-col items-center justify-center">
            <label
              htmlFor="image-upload"
              className={`relative cursor-pointer ${
                isLoading ? 'opacity-50 pointer-events-none' : ''
              }`}
            >
              <div className="flex flex-col items-center justify-center w-full p-6 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-500 transition-colors">
                <Upload className="w-12 h-12 text-gray-400 mb-2" />
                <span className="text-sm text-gray-500">Click to upload an image</span>
              </div>
              <input
                id="image-upload"
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleImageUpload}
                disabled={isLoading}
              />
            </label>
          </div>
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="flex items-center justify-center p-4">
            <Loader className="w-8 h-8 text-blue-500 animate-spin" />
            <span className="ml-2 text-gray-600">Processing image...</span>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-8">
            {error}
          </div>
        )}

        {/* Results */}
        {results && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4 text-gray-800">Mask</h2>
              <img
                src={results.mask}
                alt="Mask Result"
                className="w-full h-auto rounded-lg"
              />
            </div>
            <div className="bg-white p-4 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4 text-gray-800">Overlay</h2>
              <img
                src={results.overlay}
                alt="Overlay Result"
                className="w-full h-auto rounded-lg"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;