import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronUp, Activity, AlertCircle, CheckCircle2 } from 'lucide-react';
interface ExpandablePanelProps {
  title: string;
  count?: number;
  type: 'success' | 'warning' | 'neutral';
  children: React.ReactNode;
  defaultExpanded?: boolean;
}
export function ExpandablePanel({
  title,
  count,
  type,
  children,
  defaultExpanded = false
}: ExpandablePanelProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const getIcon = () => {
    switch (type) {
      case 'success':
        return <CheckCircle2 className="w-5 h-5 text-green-400" />;
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-amber-400" />;
      default:
        return <Activity className="w-5 h-5 text-blue-400" />;
    }
  };
  const getBorderColor = () => {
    switch (type) {
      case 'success':
        return 'border-green-500/20 hover:border-green-500/40';
      case 'warning':
        return 'border-amber-500/20 hover:border-amber-500/40';
      default:
        return 'border-blue-500/20 hover:border-blue-500/40';
    }
  };
  return <div className={`w-full bg-[#22262e] border ${getBorderColor()} rounded-lg overflow-hidden transition-colors duration-300 mb-4`}>
      <button onClick={() => setIsExpanded(!isExpanded)} className="w-full flex items-center justify-between p-4 bg-[#2a2f38]/50 hover:bg-[#2a2f38] transition-colors">
        <div className="flex items-center gap-3">
          {getIcon()}
          <span className="font-medium text-gray-200 tracking-wide text-sm uppercase">
            {title}
          </span>
          {count !== undefined && <span className="px-2 py-0.5 bg-gray-800 rounded text-xs font-mono-data text-gray-400 border border-gray-700">
              {count}
            </span>}
        </div>

        {isExpanded ? <ChevronUp className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
      </button>

      <AnimatePresence>
        {isExpanded && <motion.div initial={{
        height: 0,
        opacity: 0
      }} animate={{
        height: 'auto',
        opacity: 1
      }} exit={{
        height: 0,
        opacity: 0
      }} transition={{
        duration: 0.3,
        ease: 'easeInOut'
      }} className="overflow-hidden">
            <div className="p-4 border-t border-gray-800 bg-[#1f2329]">
              {children}
            </div>
          </motion.div>}
      </AnimatePresence>
    </div>;
}