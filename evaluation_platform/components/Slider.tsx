import React from 'react';

interface SliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void; // This function will be used to update the parent component's state
}

const Slider: React.FC<SliderProps> = ({ label, value, onChange }) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(Number(e.target.value)); // Call the onChange function passed from the parent component
  };

  return (
    <div className="flex flex-col items-start justify-between w-full">
      <label className="text-sm text-gray-600 mb-2">{label}</label>
      <input
        type="range"
        min={1}
        max={5}
        value={value}
        className="w-full"
        onChange={handleChange}
      />
      <span className="ml-auto mt-1">{value}</span>
    </div>
  );
};

export default Slider;