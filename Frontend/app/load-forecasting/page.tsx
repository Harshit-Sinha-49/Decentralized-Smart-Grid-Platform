'use client'
import { useState } from "react";
import axios from "axios";
import Link from 'next/link';

//const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;


export default function LoadForecasting() {
  const [trainFile, setTrainFile] = useState<File | null>(null);
  const [predictFile, setPredictFile] = useState<File | null>(null);
  const [trainMessage, setTrainMessage] = useState("");
  const [predictMessage, setPredictMessage] = useState("");
  const [predictions, setPredictions] = useState<number[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleTrainFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setTrainFile(event.target.files[0]);
    }
  };
  
  const handlePredictFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setPredictFile(event.target.files[0]);
    }
  };

  const handleTrainLocal = async () => {
    if (!trainFile) {
      setTrainMessage("Please select a training file first.");
      return;
    }

    setIsTraining(true);
    setProgress(10);

    const formData = new FormData();
    formData.append("file", trainFile);

    try {
      const response = await axios.post("http://35.154.18.210/train_local", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) { 
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setProgress(percentCompleted);
          }
        },
      });

      setProgress(100);
      setTrainMessage(`Training Complete!\n R2 Score: ${response.data["R2 Score"]},\n MSE: ${response.data["MSE"]}`);
    } catch (error) {
      setTrainMessage("Training failed. Please try again.");
      console.error(error);
    } finally {
      setTimeout(() => {
        setIsTraining(false);
        setProgress(0);
      }, 1500);
    }
  };

  const handleGetPredictions = async () => {
    if (!predictFile) {
      setPredictMessage("Please select a prediction file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", predictFile);

    try {
      const response = await axios.post("http://35.154.18.210/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPredictions(response.data.predictions);
      setPredictMessage("Predictions fetched successfully.");
    } catch (error) {
      setPredictMessage("Prediction failed. Please try again.");
      console.error(error);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-5">
      <div className="bg-white p-6 rounded-lg shadow-md w-full max-w-md">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-2xl font-bold">Energy Demand Forecasting</h1>
          <Link href="/" className="text-blue-500 hover:underline">
            Home
          </Link>
        </div>
        
        <h2 className="text-lg font-semibold">Upload Training Dataset:</h2>
        <input type="file" onChange={handleTrainFileChange} className="mb-4 w-full border p-2 rounded" />

        <button 
          onClick={handleTrainLocal} 
          className="w-full bg-blue-500 text-white p-2 rounded mt-2 hover:bg-blue-600 hover:cursor-pointer"
          disabled={isTraining}
        >
          {isTraining ? "Training..." : "Train Local Model"}
        </button>

        {isTraining && (
          <div className="w-full bg-gray-200 rounded-full h-2.5 mt-3">
            <div 
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-300 ease-in-out"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        )}

        {trainMessage && (
          <div className="mt-4 bg-gray-200 p-4 rounded">
            <h3 className="text-lg font-semibold">Training Status:</h3>
            <p className="text-gray-800">{trainMessage}</p>
          </div>
        )}

        <h2 className="text-lg font-semibold mt-6">Upload Prediction Dataset:</h2>
        <input type="file" onChange={handlePredictFileChange} className="mb-4 w-full border p-2 rounded" />

        <button 
          onClick={handleGetPredictions} 
          className="w-full bg-green-500 text-white p-2 rounded mt-2 hover:bg-green-600 hover:cursor-pointer"
          disabled={isTraining}
        >
          Get Predictions
        </button>

        {predictMessage && (
          <div className="mt-4 bg-gray-200 p-4 rounded">
            <h3 className="text-lg font-semibold">Prediction Status:</h3>
            <p className="text-gray-800">{predictMessage}</p>
          </div>
        )}

        {predictions.length > 0 && (
          <div className="mt-4 bg-gray-200 p-4 rounded">
            <h3 className="text-lg font-semibold">Predictions:</h3>
            <ul className="mt-2">
              {predictions.slice(0, 5).map((pred, index) => (
                <li key={index} className="text-gray-800">{pred.toFixed(2)} kW</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
