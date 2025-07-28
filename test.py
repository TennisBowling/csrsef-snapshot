#!/usr/bin/env python3

import argparse

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

def main():
    parser = argparse.ArgumentParser(description='Convert temperatures between Celsius and Fahrenheit.')
    parser.add_argument('value', type=float, help='The temperature value to convert')
    parser.add_argument('-c', '--celsius', action='store_true', help='Convert from Celsius to Fahrenheit')
    parser.add_argument('-f', '--fahrenheit', action='store_true', help='Convert from Fahrenheit to Celsius')
    
    args = parser.parse_args()
    
    if args.celsius:
        fahrenheit = celsius_to_fahrenheit(args.value)
        print(f'{args.value} °C is equal to {fahrenheit:.2f} °F')
    elif args.fahrenheit:
        celsius = fahrenheit_to_celsius(args.value)
        print(f'{args.value} °F is equal to {celsius:.2f} °C')
    else:
        print("Please specify a conversion: use -c for Celsius to Fahrenheit or -f for Fahrenheit to Celsius.")
        parser.print_help()

if __name__ == '__main__':
    main()