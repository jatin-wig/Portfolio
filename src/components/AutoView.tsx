import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Play, Pause, RotateCcw, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { motion, AnimatePresence } from 'framer-motion';
import { SectionHighlight, sectionDescriptions } from './SectionHighlight';
import AutoCursor from './AutoCursor';

interface AutoViewProps {
  onSectionChange?: (sectionIndex: number) => void;
  showFullUI?: boolean;
  onTourComplete?: () => void;
  onTourStop?: () => void;
  autoStart?: boolean;
}

// Dynamic sections generation function
const generateSections = () => {
  const baseSections = [
    { id: 'hero', name: 'Home', duration: 6000, type: 'section' },
    { id: 'superpowers', name: 'Skills', duration: 7000, type: 'section' },
    { id: 'why-choose-me', name: 'Why Choose Me', duration: 6000, type: 'section' },
    { id: 'projects', name: 'Projects Overview', duration: 8000, type: 'section' },
  ];

  const endSections = [
    { id: 'publications', name: 'Research & Publications', duration: 7000, type: 'section' },
    { id: 'experience', name: 'Experience', duration: 8000, type: 'section' },
    { id: 'volunteering', name: 'Volunteering', duration: 6000, type: 'section' },
    { id: 'education', name: 'Education', duration: 6000, type: 'section' },
    { id: 'resume', name: 'Resume', duration: 6000, type: 'section' },
    { id: 'contact', name: 'Contact', duration: 6000, type: 'section' }
  ];

  return [...baseSections, ...endSections];
};

const speedOptions = [
  { label: 'Slow', multiplier: 1.5 },
  { label: 'Normal', multiplier: 1 },
  { label: 'Fast', multiplier: 0.7 },
  { label: 'Very Fast', multiplier: 0.5 }
];

const AutoView: React.FC<AutoViewProps> = ({ 
  onSectionChange, 
  showFullUI = true, 
  onTourComplete,
  onTourStop,
  autoStart = false
}) => {
  const [isPlaying, setIsPlaying] = useState(autoStart);
  const [isPaused, setIsPaused] = useState(false);
  const [currentSection, setCurrentSection] = useState(0);
  const [progress, setProgress] = useState(0);
  const [showHighlight, setShowHighlight] = useState(false);
  const [cursorPosition, setCursorPosition] = useState({ x: 0, y: 0 });
  const [showCursor, setShowCursor] = useState(false);
  const [isClicking, setIsClicking] = useState(false);
  const [speedIndex, setSpeedIndex] = useState(1); // Normal speed by default
  const [showSpeedMenu, setShowSpeedMenu] = useState(false);

  const currentSpeed = speedOptions[speedIndex];
  
  // Memoize sections to avoid regenerating on every render
  const sections = useMemo(() => generateSections(), []);

  const simulateClick = useCallback((element: Element) => {
    const rect = element.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;
    
    setCursorPosition({ x, y });
    setShowCursor(true);
    
    setTimeout(() => {
      setIsClicking(true);
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      
      // Simulate actual click for project cards
      if (element.hasAttribute('data-project-id')) {
        (element as HTMLElement).click();
      }
      
      setTimeout(() => {
        setIsClicking(false);
        setTimeout(() => setShowCursor(false), 500);
      }, 200);
    }, 1000);
  }, []);

  const scrollToSection = useCallback((sectionId: string) => {
    if (sectionId.startsWith('project-')) {
      // For individual projects, scroll to projects section first
      const projectsSection = document.getElementById('projects');
      if (projectsSection) {
        projectsSection.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'start',
          inline: 'nearest'
        });
        
        // Then find and click the specific project card
        setTimeout(() => {
          const projectId = parseInt(sectionId.split('-')[1]);
          const projectCard = document.querySelector(`[data-project-id="${projectId}"]`);
          if (projectCard) {
            simulateClick(projectCard);
          }
        }, 1000);
      }
    } else {
      const element = document.getElementById(sectionId);
      if (element) {
        element.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'start',
          inline: 'nearest'
        });
      }
    }
  }, [simulateClick]);

  const nextSection = useCallback(() => {
    if (currentSection < sections.length - 1) {
      const newSection = currentSection + 1;
      setCurrentSection(newSection);
      scrollToSection(sections[newSection].id);
      onSectionChange?.(newSection);
      return true;
    }
    return false;
  }, [currentSection, scrollToSection, onSectionChange]);

  const resetTour = useCallback(() => {
    setCurrentSection(0);
    setProgress(0);
    setShowHighlight(false);
    setShowCursor(false);
    scrollToSection(sections[0].id);
    onSectionChange?.(0);
  }, [scrollToSection, onSectionChange]);

  const stopTour = useCallback(() => {
    setIsPlaying(false);
    setIsPaused(false);
    setShowHighlight(false);
    setShowCursor(false);
    onTourStop?.();
  }, [onTourStop]);

  const pauseTour = useCallback(() => {
    setIsPaused(true);
    setShowHighlight(false);
    setShowCursor(false);
  }, []);

  const resumeTour = useCallback(() => {
    setIsPaused(false);
    scrollToSection(sections[currentSection].id);
    setShowHighlight(true);
  }, [currentSection, scrollToSection]);

  const toggleAutoView = useCallback(() => {
    if (!isPlaying) {
      if (currentSection === sections.length - 1) {
        resetTour();
      }
      setIsPaused(false);
      scrollToSection(sections[currentSection].id);
      setShowHighlight(true);
      setIsPlaying(true);
    }
  }, [isPlaying, currentSection, scrollToSection, resetTour]);

  // Auto-start effect
  useEffect(() => {
    if (autoStart) {
      scrollToSection(sections[0].id);
      setShowHighlight(true);
    }
  }, [autoStart, scrollToSection]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    let progressInterval: NodeJS.Timeout;
    let highlightTimeout: NodeJS.Timeout;

    if (isPlaying && !isPaused) {
      const baseDuration = sections[currentSection].duration;
      const currentDuration = baseDuration * currentSpeed.multiplier;
      
      // Show highlight at the beginning of each section
      setShowHighlight(true);
      highlightTimeout = setTimeout(() => {
        setShowHighlight(false);
      }, Math.min(currentDuration - 500, 4000));
      
      // Progress bar animation
      setProgress(0);
      progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) return 100;
          return prev + (100 / (currentDuration / 50));
        });
      }, 50);

      // Section transition
      interval = setTimeout(() => {
        const hasNext = nextSection();
        if (!hasNext) {
          setIsPlaying(false);
          setProgress(0);
          setShowHighlight(false);
          setShowCursor(false);
          onTourComplete?.();
        }
      }, currentDuration);
    }

    return () => {
      if (interval) clearTimeout(interval);
      if (progressInterval) clearInterval(progressInterval);
      if (highlightTimeout) clearTimeout(highlightTimeout);
    };
  }, [isPlaying, isPaused, currentSection, nextSection, currentSpeed.multiplier, onTourComplete]);

  const currentSectionName = sections[currentSection].name;
  const currentDescription = sectionDescriptions[currentSectionName] || `Exploring ${currentSectionName}`;

  if (!showFullUI) {
    return null;
  }

  return (
    <>
      <AutoCursor 
        isVisible={showCursor} 
        position={cursorPosition} 
        isClicking={isClicking}
      />
      
      <div className="fixed top-4 right-4 z-50">
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-white/95 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200 p-4 min-w-[320px]"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-gray-800 flex items-center gap-2">
              🚀 Auto Portfolio Tour
            </h3>
            <div className="flex gap-2">
              <div className="relative">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowSpeedMenu(!showSpeedMenu)}
                  className="p-2"
                >
                  <Settings className="h-4 w-4" />
                </Button>
                <AnimatePresence>
                  {showSpeedMenu && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      className="absolute top-10 right-0 bg-white border border-gray-200 rounded-lg shadow-lg p-2 min-w-[100px]"
                    >
                      {speedOptions.map((speed, index) => (
                        <button
                          key={speed.label}
                          onClick={() => {
                            setSpeedIndex(index);
                            setShowSpeedMenu(false);
                          }}
                          className={`block w-full text-left px-3 py-1 rounded text-sm hover:bg-gray-100 ${
                            speedIndex === index ? 'bg-brand-purple text-white' : ''
                          }`}
                        >
                          {speed.label}
                        </button>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={resetTour}
                disabled={isPlaying && !isPaused}
                className="p-2"
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
              {isPlaying && !isPaused && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={pauseTour}
                  className="flex items-center gap-2"
                >
                  <Pause className="h-4 w-4" />
                  Pause
                </Button>
              )}
              {isPlaying && isPaused && (
                <Button
                  variant="default"
                  size="sm"
                  onClick={resumeTour}
                  className="flex items-center gap-2"
                >
                  <Play className="h-4 w-4" />
                  Resume
                </Button>
              )}
              {!isPlaying && (
                <Button
                  variant="default"
                  size="sm"
                  onClick={toggleAutoView}
                  className="flex items-center gap-2"
                >
                  <Play className="h-4 w-4" />
                  Start Tour
                </Button>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={stopTour}
                className="flex items-center gap-2"
              >
                Stop
              </Button>
            </div>
          </div>

          <AnimatePresence mode="wait">
            {(isPlaying || isPaused) && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="space-y-3"
              >
                <div className="text-sm text-gray-600">
                  <div className="flex justify-between items-center mb-1">
                    <span>Current: {currentSectionName}</span>
                    <span className="text-xs">{currentSection + 1}/{sections.length}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <motion.div
                      className="bg-gradient-to-r from-brand-purple to-brand-vivid-purple h-2 rounded-full"
                      style={{ width: `${progress}%` }}
                      initial={{ width: 0 }}
                      animate={{ width: `${progress}%` }}
                      transition={{ duration: 0.1 }}
                    />
                  </div>
                </div>
                
                <div className="flex justify-between items-center text-xs text-gray-500">
                  <span>Speed: {currentSpeed.label}</span>
                  <span>
                    {isPaused ? 'Paused' : 
                     currentSection < sections.length - 1 
                      ? `Next: ${sections[currentSection + 1].name}`
                      : 'Tour Complete'
                    }
                  </span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {!isPlaying && currentSection === sections.length - 1 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-sm text-gray-600 text-center py-2"
            >
              Tour completed! 🎉
            </motion.div>
          )}
        </motion.div>
      </div>

      <SectionHighlight
        isVisible={showHighlight && isPlaying && !isPaused}
        title={currentSectionName}
        description={currentDescription}
      />
    </>
  );
};

export default AutoView;