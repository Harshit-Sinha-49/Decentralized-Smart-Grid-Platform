'use client'
import Link from 'next/link';
import Image from 'next/image';

export default function Home() {
  return (
    <div className="relative flex flex-col items-center justify-center min-h-screen bg-gray-100 p-5 overflow-hidden">
      {/* Background Image */}
      <div className="absolute inset-0 z-0 opacity-60">
        <Image 
          src="/Images/smart-grid-background.jpg" 
          alt="Smart Grid Network" 
          layout="fill" 
          objectFit="cover" 
          quality={75}
          className="filter brightness-50 contrast-125"
        />
      </div>
      
      <div className="relative bg-white/90 p-12 rounded-xl shadow-2xl w-full max-w-4xl text-center z-10">
        <h1 className="text-4xl font-bold mb-8 text-gray-800">Decentralized Smart Grid Energy Intelligence Platform</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-blue-100 p-7 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-blue-800">Decentralized Energy Demand Forecasting</h2>
            <p className="text-gray-700 mb-5">
              Predict energy demand using an MLP regressor for accurate and reliable forecasting. 
              Upload training and prediction datasets to generate accurate forecasts.
            </p>
            <Link href="/load-forecasting" className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition whitespace-nowrap">
              Access Load Forecasting
            </Link>
          </div>
          
          <div className="bg-green-100 p-7 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-green-800">Decentralized Transformer Fault Detection</h2>
            <p className="text-gray-700 mb-5">
              Utilize Isolation Forest (Machine Learning Technique) to detect potential faults in transformers. 
              Train models and predict fault probabilities using historical data.
            </p>
            <Link href="/transformer-fault" className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 transition">
              Access Fault Detection
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}