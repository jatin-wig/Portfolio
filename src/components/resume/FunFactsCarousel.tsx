
import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { 
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselApi,
} from "@/components/ui/carousel";
import funFacts from './FunFactsData';

const FunFactsCarousel = () => {
  const [currentFactIndex, setCurrentFactIndex] = useState(0);
  const [api, setApi] = useState<CarouselApi>();

  const showPreviousFact = useCallback(() => {
    api?.scrollPrev();
  }, [api]);

  const showNextFact = useCallback(() => {
    api?.scrollNext();
  }, [api]);

  const handleCarouselSelect = useCallback(() => {
    if (!api) return;
    setCurrentFactIndex(api.selectedScrollSnap());
  }, [api]);

  React.useEffect(() => {
    if (!api) return;
    
    api.on("select", handleCarouselSelect);
    
    return () => {
      api.off("select", handleCarouselSelect);
    };
  }, [api, handleCarouselSelect]);

  return (
    <div className="relative overflow-hidden pb-12">
      <Carousel
        opts={{
          loop: true,
          align: "center",
        }}
        className="w-full -mt-6"
        setApi={setApi}
      >
        <CarouselContent>
          {funFacts.map((fact, index) => (
            <CarouselItem key={index} className="w-full pl-4">
              <AnimatePresence mode="wait">
                <motion.div 
                  key={index}
                  className="fun-fact-card p-6 bg-white rounded-lg shadow-xl h-full border-t-4 border-brand-purple"
                  initial={{ opacity: 0, y: 20, scale: 0.95 }}
                  animate={{ 
                    opacity: 1, 
                    y: 0, 
                    scale: 1,
                    transition: {
                      type: "spring",
                      stiffness: 300,
                      damping: 20,
                      duration: 0.7
                    }
                  }}
                  exit={{ 
                    opacity: 0, 
                    y: -20, 
                    scale: 0.95,
                    transition: {
                      duration: 0.3
                    }
                  }}
                  whileHover={{ 
                    y: -5,
                    boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)"
                  }}
                  transition={{ 
                    type: "spring", 
                    stiffness: 300, 
                    damping: 20,
                    duration: 0.7 
                  }}
                >
                  <div className="flex items-center gap-4 mb-4">
                    <motion.div 
                      className="w-12 h-12 bg-gradient-to-br from-brand-light-purple to-brand-purple rounded-full flex items-center justify-center text-2xl shadow-md"
                      animate={{ 
                        scale: [1, 1.1, 1],
                        rotate: [0, 5, -5, 0]
                      }}
                      transition={{
                        duration: 2,
                        ease: "easeInOut",
                        times: [0, 0.2, 0.5, 1],
                        repeat: Infinity,
                        repeatDelay: 1
                      }}
                    >
                      {fact.icon}
                    </motion.div>
                    <h3 className="text-xl font-bold">{fact.title}</h3>
                  </div>
                  <p className="text-gray-600">{fact.text}</p>
                </motion.div>
              </AnimatePresence>
            </CarouselItem>
          ))}
        </CarouselContent>
        
        <div className="mt-8 flex justify-between items-center">
          <motion.button 
            onClick={showPreviousFact}
            className="flex items-center justify-center w-10 h-10 rounded-full bg-white shadow-md hover:shadow-lg transition-all duration-300 text-brand-purple"
            whileHover={{ scale: 1.1, backgroundColor: "#f9f5ff" }}
            whileTap={{ scale: 0.9 }}
          >
            <ChevronLeft className="h-6 w-6" />
          </motion.button>
          
          <div className="flex gap-2">
            {funFacts.map((_, index) => (
              <motion.button 
                key={index}
                className={`h-2 rounded-full transition-all duration-300 ${
                  index === currentFactIndex ? "bg-brand-purple w-8" : "bg-gray-300 w-2"
                }`}
                onClick={() => api?.scrollTo(index)}
                whileHover={{ scale: 1.2 }}
                whileTap={{ scale: 0.9 }}
              />
            ))}
          </div>
          
          <motion.button 
            onClick={showNextFact}
            className="flex items-center justify-center w-10 h-10 rounded-full bg-white shadow-md hover:shadow-lg transition-all duration-300 text-brand-purple"
            whileHover={{ scale: 1.1, backgroundColor: "#f9f5ff" }}
            whileTap={{ scale: 0.9 }}
          >
            <ChevronRight className="h-6 w-6" />
          </motion.button>
        </div>
      </Carousel>
      
      <div className="absolute top-2 right-2 bg-brand-light-purple text-brand-purple px-3 py-1 rounded-full font-medium text-sm">
        {currentFactIndex + 1}/{funFacts.length}
      </div>
    </div>
  );
};

export default FunFactsCarousel;