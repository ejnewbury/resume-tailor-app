import React from 'react';
import { motion } from 'framer-motion';
interface CircularProgressProps {
  percentage: number;
  size?: number;
  strokeWidth?: number;
  label?: string;
}
export function CircularProgress({
  percentage,
  size = 200,
  strokeWidth = 12,
  label = 'MATCH'
}: CircularProgressProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - percentage / 100 * circumference;
  return <div className="relative flex flex-col items-center justify-center" style={{
    width: size,
    height: size
  }}>
      {/* Glow Effect Background */}
      <div className="absolute inset-0 rounded-full bg-blue-500/5 blur-2xl" style={{
      transform: 'scale(0.9)'
    }} />

      <svg width={size} height={size} className="transform -rotate-90 relative z-10">
        {/* Background Circle */}
        <circle cx={size / 2} cy={size / 2} r={radius} stroke="#2d3748" strokeWidth={strokeWidth} fill="transparent" />
        {/* Progress Circle */}
        <motion.circle cx={size / 2} cy={size / 2} r={radius} stroke="#3b82f6" strokeWidth={strokeWidth} fill="transparent" strokeLinecap="round" strokeDasharray={circumference} initial={{
        strokeDashoffset: circumference
      }} animate={{
        strokeDashoffset: offset
      }} transition={{
        duration: 1.5,
        ease: 'easeOut'
      }} style={{
        filter: 'drop-shadow(0 0 6px rgba(59, 130, 246, 0.5))'
      }} />
      </svg>

      {/* Center Text */}
      <div className="absolute inset-0 flex flex-col items-center justify-center z-20">
        <motion.span className="text-5xl font-bold font-mono-data text-white tracking-tighter" initial={{
        opacity: 0,
        y: 10
      }} animate={{
        opacity: 1,
        y: 0
      }} transition={{
        delay: 0.5,
        duration: 0.5
      }}>
          {Math.round(percentage)}%
        </motion.span>
        <span className="text-xs text-blue-400 font-medium tracking-widest mt-1 uppercase">
          {label}
        </span>
      </div>
    </div>;
}